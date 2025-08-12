# src/training/steps/step9_tactician_specialist_training.py

import asyncio
import json
import os
import pickle
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors
from src.utils.warning_symbols import (
    error,
    failed,
)
from src.training.steps.unified_data_loader import get_unified_data_loader


class TacticianSpecialistTrainingStep:
    """Step 9: Tactician Specialist Models Training (LightGBM + Calibrated Logistic Regression)."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = system_logger
        self.models = {}

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="tactician specialist training step initialization",
    )
    async def initialize(self) -> None:
        """Initialize the tactician specialist training step."""
        self.logger.info("Initializing Tactician Specialist Training Step...")
        self.logger.info(
            "Tactician Specialist Training Step initialized successfully",
        )

    @handle_errors(
        exceptions=(Exception,),
        default_return={"status": "FAILED", "error": "Execution failed"},
        context="tactician specialist training step execution",
    )
    async def execute(
        self,
        training_input: dict[str, Any],
        pipeline_state: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute tactician specialist models training.

        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state

        Returns:
            Dict containing training results
        """
        try:
            self.logger.info("🔄 Executing Tactician Specialist Training...")

            # Extract parameters
            symbol = training_input.get("symbol", "ETHUSDT")
            exchange = training_input.get("exchange", "BINANCE")
            data_dir = training_input.get("data_dir", "data/training")

            # Load tactician labeled data
            labeled_data_dir = f"{data_dir}/tactician_labeled_data"
            labeled_file_parquet = (
                f"{labeled_data_dir}/{exchange}_{symbol}_tactician_labeled.parquet"
            )
            labeled_file_pickle = (
                f"{labeled_data_dir}/{exchange}_{symbol}_tactician_labeled.pkl"
            )

            if os.path.exists(labeled_file_parquet):
                # Prefer partitioned dataset scan from labeled store if available
                try:
                    from src.training.enhanced_training_manager_optimized import (
                        ParquetDatasetManager,
                    )

                    pdm = ParquetDatasetManager(logger=self.logger)
                    part_base = os.path.join(data_dir, "parquet", "labeled")
                    if os.path.isdir(part_base):
                        filters = [
                            ("exchange", "==", exchange),
                            ("symbol", "==", symbol),
                            ("timeframe", "==", training_input.get("timeframe", "1m")),
                            ("split", "==", "train"),
                        ]
                        # Reader shortcut: prefer materialized projection if available
                        feat_cols = training_input.get(
                            "model_feature_columns"
                        ) or training_input.get("feature_columns")
                        label_col = training_input.get("label_column", "label")
                        proj_base = os.path.join(
                            "data_cache",
                            "parquet",
                            f"proj_features_{training_input.get('model_name', 'default')}",
                        )
                        if (
                            isinstance(feat_cols, list)
                            and len(feat_cols) > 0
                            and os.path.isdir(proj_base)
                        ):
                            proj_filters = [
                                ("exchange", "==", exchange),
                                ("symbol", "==", symbol),
                                (
                                    "timeframe",
                                    "==",
                                    training_input.get("timeframe", "1m"),
                                ),
                                ("split", "==", "train"),
                            ]
                            cols = ["timestamp", *feat_cols, label_col]
                            labeled_data = pdm.cached_projection(
                                base_dir=proj_base,
                                filters=proj_filters,
                                columns=cols,
                                cache_dir="data_cache/projections",
                                cache_key_prefix=f"proj_features_{training_input.get('model_name','default')}_{exchange}_{symbol}_{training_input.get('timeframe','1m')}_train",
                                snapshot_version="v1",
                                ttl_seconds=3600,
                                batch_size=131072,
                            )
                        else:
                            cache_key = f"labeled_{exchange}_{symbol}_{training_input.get('timeframe','1m')}_train"
                            labeled_data = pdm.cached_projection(
                                base_dir=part_base,
                                filters=filters,
                                columns=[],
                                cache_dir="data_cache/projections",
                                cache_key_prefix=cache_key,
                                snapshot_version="v1",
                                ttl_seconds=3600,
                                batch_size=131072,
                                arrow_transform=lambda tbl: (
                                    (lambda _t: _t)(
                                        (
                                            lambda _pa, pc: (
                                                _t := tbl,
                                                (
                                                    _t := _t.set_column(
                                                        _t.schema.get_field_index(
                                                            "timestamp"
                                                        ),
                                                        "timestamp",
                                                        pc.cast(
                                                            _t.column("timestamp"),
                                                            _pa.int64(),
                                                        ),
                                                    )
                                                )
                                                if (
                                                    "timestamp" in _t.schema.names
                                                    and not _pa.types.is_int64(
                                                        _t.schema.field(
                                                            "timestamp"
                                                        ).type
                                                    )
                                                )
                                                else None,
                                                _t,
                                            )
                                        )(
                                            __import__("pyarrow"),
                                            __import__("pyarrow.compute"),
                                        )
                                    )
                                ),
                            )
                    else:
                        try:
                            feat_cols = training_input.get(
                                "model_feature_columns"
                            ) or training_input.get("feature_columns")
                            label_col = training_input.get("label_column", "label")
                            from src.utils.logger import (
                                log_io_operation,
                                log_dataframe_overview,
                            )

                            if isinstance(feat_cols, list) and len(feat_cols) > 0:
                                with log_io_operation(
                                    self.logger,
                                    "read_parquet",
                                    labeled_file_parquet,
                                    columns=True,
                                ):
                                    labeled_data = pd.read_parquet(
                                        labeled_file_parquet,
                                        columns=["timestamp", *feat_cols, label_col],
                                    )
                            else:
                                with log_io_operation(
                                    self.logger, "read_parquet", labeled_file_parquet
                                ):
                                    labeled_data = pd.read_parquet(labeled_file_parquet)
                            try:
                                log_dataframe_overview(
                                    self.logger, labeled_data, name="labeled_data"
                                )
                            except Exception:
                                pass
                        except Exception:
                            with log_io_operation(
                                self.logger, "read_parquet", labeled_file_parquet
                            ):
                                labeled_data = pd.read_parquet(labeled_file_parquet)
                except Exception:
                    try:
                        feat_cols = training_input.get(
                            "model_feature_columns"
                        ) or training_input.get("feature_columns")
                        label_col = training_input.get("label_column", "label")
                        from src.utils.logger import log_io_operation

                        if isinstance(feat_cols, list) and len(feat_cols) > 0:
                            with log_io_operation(
                                self.logger,
                                "read_parquet",
                                labeled_file_parquet,
                                columns=True,
                            ):
                                labeled_data = pd.read_parquet(
                                    labeled_file_parquet,
                                    columns=["timestamp", *feat_cols, label_col],
                                )
                        else:
                            with log_io_operation(
                                self.logger, "read_parquet", labeled_file_parquet
                            ):
                                labeled_data = pd.read_parquet(labeled_file_parquet)
                    except Exception:
                        with log_io_operation(
                            self.logger, "read_parquet", labeled_file_parquet
                        ):
                            labeled_data = pd.read_parquet(labeled_file_parquet)
            elif os.path.exists(labeled_file_pickle):
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
            try:
                shape = getattr(labeled_data, "shape", None)
                self.logger.info(f"Loaded tactician labeled data: shape={shape}")
                if (
                    isinstance(labeled_data, pd.DataFrame)
                    and "tactician_label" in labeled_data.columns
                ):
                    self.logger.info(
                        f"Label distribution: {labeled_data['tactician_label'].value_counts().to_dict()}",
                    )
            except Exception:
                pass

            # Use labeled_data downstream
            # Mandatory: augment features with SR model signals
            try:
                # Load SR models
                sr_models_dir = os.path.join(data_dir, "enhanced_analyst_models", "SR")
                if not os.path.isdir(sr_models_dir):
                    sr_models_dir = os.path.join(data_dir, "analyst_models", "SR")
                sr_models: dict[str, Any] = {}
                if os.path.isdir(sr_models_dir):
                    for mf in os.listdir(sr_models_dir):
                        if mf.endswith((".pkl", ".joblib")):
                            mp = os.path.join(sr_models_dir, mf)
                            try:
                                if mf.endswith(".joblib"):
                                    import joblib
                                    sr_models[mf.replace(".joblib", "")] = joblib.load(mp)
                                else:
                                    with open(mp, "rb") as f:
                                        sr_models[mf.replace(".pkl", "")] = pickle.load(f)
                            except Exception:
                                continue
                # Compute SR predictions as features
                if sr_models:
                    def _ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
                        obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
                        if obj_cols:
                            df = df.drop(columns=obj_cols)
                        dt_cols = df.select_dtypes(include=["datetime", "datetime64", "datetime64[ns]"]).columns.tolist()
                        if dt_cols:
                            df = df.drop(columns=dt_cols)
                        return df
                    X_all = _ensure_numeric(labeled_data.drop(columns=[c for c in ["label"] if c in labeled_data.columns], errors="ignore")).select_dtypes(include=[np.number])
                    for name, model in sr_models.items():
                        try:
                            # Some models may require matching columns; use intersection
                            cols = [c for c in getattr(model, "feature_names_in_", X_all.columns) if c in X_all.columns]
                            if not cols:
                                continue
                            proba = model.predict_proba(X_all[cols])
                            if proba.shape[1] >= 2:
                                labeled_data[f"sr_sig_{name}_p1"] = proba[:, 1]
                                labeled_data[f"sr_sig_{name}_p0"] = proba[:, 0]
                            else:
                                labeled_data[f"sr_sig_{name}_p1"] = proba.reshape(-1)
                        except Exception:
                            continue
                    self.logger.info(f"✅ Augmented tactician features with {len(sr_models)} SR model signals")
                else:
                    self.logger.warning("No SR models found; tactician SR augmentation skipped")
            except Exception as _e:
                self.logger.warning(f"Tactician SR signal augmentation failed: {_e}")

            # Optionally drop raw S/R features to reduce redundancy (keep SR signals)
            try:
                drop_raw_sr = bool(self.config.get("tactician", {}).get("drop_raw_sr_features", False))
                if drop_raw_sr:
                    sr_raw_cols = [
                        "dist_to_support_pct",
                        "dist_to_resistance_pct",
                        "sr_zone_position",
                        "nearest_support_center",
                        "nearest_resistance_center",
                        "nearest_support_score",
                        "nearest_resistance_score",
                        "nearest_support_band_pct",
                        "nearest_resistance_band_pct",
                        "sr_breakout_up",
                        "sr_breakout_down",
                        "sr_bounce_up",
                        "sr_bounce_down",
                        "sr_touch",
                        "sr_breakout_score",
                        "sr_bounce_score",
                    ]
                    # Do not drop SR model signal columns prefixed with 'sr_sig_' or strength predictions 'sr_pred_'
                    present = [c for c in sr_raw_cols if c in labeled_data.columns]
                    if present:
                        labeled_data = labeled_data.drop(columns=present)
                        self.logger.info(f"🔧 Dropped raw SR features from tactician training: {present}")
            except Exception as _ed:
                self.logger.warning(f"Unable to drop raw SR features: {_ed}")

            # Train tactician specialist models
            training_results = await self._train_tactician_models(
                labeled_data,
                training_input,
                pipeline_state,
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

        except Exception as e:
            self.print(error("❌ Error in Tactician Specialist Training: {e}"))
            return {"status": "FAILED", "error": str(e), "duration": 0.0}

    async def _train_tactician_models(
        self,
        data: pd.DataFrame,
        symbol: str,
        exchange: str,
    ) -> dict[str, Any]:
        """
        Train tactician specialist models.

        Args:
            data: Labeled data for tactician
            symbol: Trading symbol
            exchange: Exchange name

        Returns:
            Dict containing trained models
        """
        try:
            self.logger.info(
                f"Training tactician specialist models for {symbol} on {exchange}...",
            )

            # Prepare data - handle data types properly
            # Save target columns before dropping object columns
            target_columns = ["tactician_label", "regime"]
            y = data["tactician_label"].copy()

            # First, explicitly drop any datetime columns
            datetime_columns = data.select_dtypes(
                include=["datetime64[ns]", "datetime64", "datetime"],
            ).columns.tolist()
            if datetime_columns:
                self.logger.info(f"Dropping datetime columns: {datetime_columns}")
                data = data.drop(columns=datetime_columns)

            # Also drop any object columns that might contain datetime strings
            # But preserve target columns
            object_columns = data.select_dtypes(include=["object"]).columns.tolist()
            object_columns_to_drop = [
                col for col in object_columns if col not in target_columns
            ]
            if object_columns_to_drop:
                self.logger.info(f"Dropping object columns: {object_columns_to_drop}")
                data = data.drop(columns=object_columns_to_drop)

            # Get only numeric columns for features
            excluded_columns = target_columns
            numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            feature_columns = [
                col for col in numeric_columns if col not in excluded_columns
            ]

            if not feature_columns:
                self.logger.warning(
                    "No numeric feature columns found for tactician training",
                )
                # Create a simple fallback feature
                data["simple_feature"] = np.random.randn(len(data))
                feature_columns = ["simple_feature"]

            X = data[feature_columns].copy()

            # Additional safety check - ensure all columns are numeric
            for col in X.columns:
                if not pd.api.types.is_numeric_dtype(X[col]):
                    self.logger.warning(
                        f"Non-numeric column detected: {col} with dtype {X[col].dtype}",
                    )
                    X = X.drop(columns=[col])
                    feature_columns.remove(col)

            # Remove any remaining NaN values
            X = X.fillna(0)

            # Final check - ensure X is purely numeric
            if X.select_dtypes(include=[np.number]).shape[1] != X.shape[1]:
                self.print(error("Non-numeric columns still present in feature matrix"))
                # Force conversion to numeric, dropping any problematic columns
                X = X.select_dtypes(include=[np.number])

            self.logger.info(
                f"Using {len(feature_columns)} feature columns for tactician training",
            )

            # Split data for training and validation
            # ❌ REMOVED: Stratified split with shuffle (causes data leakage)
            # ✅ IMPLEMENTED: Chronological time-series split (leak-proof)
            split_point = int(len(X) * 0.8)  # 80% train, 20% test
            X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
            y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

            self.logger.info("✅ Using chronological time-series split (leak-proof)")

            # Train different model types
            models = {}

            # 1. LightGBM
            models["lightgbm"] = await self._train_lightgbm(
                X_train,
                X_test,
                y_train,
                y_test,
                symbol,
                exchange,
            )

            # 2. Calibrated Logistic Regression
            models["calibrated_logistic"] = await self._train_calibrated_logistic(
                X_train,
                X_test,
                y_train,
                y_test,
                symbol,
                exchange,
            )

            # 3. XGBoost (additional model)
            models["xgboost"] = await self._train_xgboost(
                X_train,
                X_test,
                y_train,
                y_test,
                symbol,
                exchange,
            )

            # 4. Random Forest (additional model)
            models["random_forest"] = await self._train_random_forest(
                X_train,
                X_test,
                y_train,
                y_test,
                symbol,
                exchange,
            )

            self.logger.info(f"Trained {len(models)} tactician models")

            return models

        except Exception:
            self.print(error("Error training tactician models: {e}"))
            raise

    async def _train_lightgbm(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        symbol: str,
        exchange: str,
    ) -> dict[str, Any]:
        """Train LightGBM model."""
        try:
            import lightgbm as lgb
            from sklearn.metrics import accuracy_score

            # Train model with adaptive regularization
            # Calculate adaptive regularization based on data characteristics
            n_samples, n_features = X_train.shape
            overfitting_risk = n_features / n_samples if n_samples > 0 else 1.0

            # Adaptive regularization parameters
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
                reg_alpha=reg_alpha,  # Adaptive L1 regularization
                reg_lambda=reg_lambda,  # Adaptive L2 regularization
                min_child_samples=min_child_samples,
                subsample=subsample,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
                early_stopping_rounds=50,
            )

            # Train with validation set
            eval_set = [(X_test, y_test)]
            model.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                eval_metric="logloss",
                verbose=False,
            )

            # Evaluate model
            y_pred = model.predict(X_test)
            model.predict_proba(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            # Get feature importance
            feature_importance = dict(
                zip(X_train.columns, model.feature_importances_, strict=False),
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
                    "reg_alpha": 0.1,
                    "reg_lambda": 0.1,
                },
            }

        except Exception:
            self.print(error("Error training LightGBM: {e}"))
            raise

    async def _train_calibrated_logistic(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        symbol: str,
        exchange: str,
    ) -> dict[str, Any]:
        """Train Calibrated Logistic Regression model."""
        try:
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score

            # Base logistic regression
            base_model = LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42,
                solver="liblinear",
            )

            # Calibrate the model
            calibrated_model = CalibratedClassifierCV(
                base_estimator=base_model,
                cv=5,
                method="isotonic",
            )

            # Train model
            calibrated_model.fit(X_train, y_train)

            # Evaluate model
            y_pred = calibrated_model.predict(X_test)
            calibrated_model.predict_proba(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            return {
                "model": calibrated_model,
                "accuracy": accuracy,
                "feature_importance": {},  # Logistic regression doesn't have direct feature importance
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

        except Exception:
            self.print(error("Error training Calibrated Logistic Regression: {e}"))
            raise

    async def _train_xgboost(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        symbol: str,
        exchange: str,
    ) -> dict[str, Any]:
        """Train XGBoost model."""
        try:
            import xgboost as xgb
            from sklearn.metrics import accuracy_score

            # Train model with adaptive regularization
            # Calculate adaptive regularization based on data characteristics
            n_samples, n_features = X_train.shape
            overfitting_risk = n_features / n_samples if n_samples > 0 else 1.0

            # Adaptive regularization parameters
            if overfitting_risk > 0.1:  # High overfitting risk
                reg_alpha = 0.1
                reg_lambda = 0.1
                min_child_weight = 10
                subsample = 0.7
            elif overfitting_risk > 0.05:  # Medium overfitting risk
                reg_alpha = 0.05
                reg_lambda = 0.05
                min_child_weight = 5
                subsample = 0.8
            else:  # Low overfitting risk
                reg_alpha = 0.01
                reg_lambda = 0.01
                min_child_weight = 1
                subsample = 0.9

            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                reg_alpha=reg_alpha,  # Adaptive L1 regularization
                reg_lambda=reg_lambda,  # Adaptive L2 regularization
                min_child_weight=min_child_weight,
                subsample=subsample,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="logloss",
                early_stopping_rounds=50,
                verbose=0,  # Reduce verbose output during training
            )

            # Train with validation set
            eval_set = [(X_test, y_test)]
            model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

            # Evaluate model
            y_pred = model.predict(X_test)
            model.predict_proba(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            # Get feature importance
            feature_importance = dict(
                zip(X_train.columns, model.feature_importances_, strict=False),
            )

            return {
                "model": model,
                "accuracy": accuracy,
                "feature_importance": feature_importance,
                "model_type": "XGBoost",
                "symbol": symbol,
                "exchange": exchange,
                "training_date": datetime.now().isoformat(),
                "hyperparameters": {
                    "n_estimators": 200,
                    "max_depth": 6,
                    "learning_rate": 0.05,
                    "reg_alpha": 0.1,
                    "reg_lambda": 0.1,
                },
            }

        except Exception:
            self.print(error("Error training XGBoost: {e}"))
            raise

    async def _train_random_forest(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        symbol: str,
        exchange: str,
    ) -> dict[str, Any]:
        """Train Random Forest model."""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score

            # Train model
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            )

            model.fit(X_train, y_train)

            # Evaluate model
            y_pred = model.predict(X_test)
            model.predict_proba(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            # Get feature importance
            feature_importance = dict(
                zip(X_train.columns, model.feature_importances_, strict=False),
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

        except Exception:
            self.print(error("Error training Random Forest: {e}"))
            raise


# For backward compatibility with existing step structure
async def run_step(
    symbol: str,
    exchange: str = "BINANCE",
    data_dir: str = "data/training",
    force_rerun: bool = False,
    **kwargs,
) -> bool:
    """
    Run the tactician specialist training step.

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

        pipeline_state = {}
        result = await step.execute(training_input, pipeline_state)

        return result.get("status") == "SUCCESS"

    except Exception:
        print(failed("Tactician specialist training failed: {e}"))
        return False


if __name__ == "__main__":
    # Test the step
    async def test():
        result = await run_step("ETHUSDT", "BINANCE", "data/training")
        print(f"Test result: {result}")

    asyncio.run(test())
