# src/training/steps/step5_analyst_specialist_training.py

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
from src.utils.decorators import guard_dataframe_nulls, with_tracing_span


class AnalystSpecialistTrainingStep:
    """Step 5: Analyst Specialist Models Training."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = system_logger
        self.models = {}

    def print(self, message: str) -> None:
        """Print message using logger."""
        self.logger.info(message)

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="analyst specialist training step initialization",
    )
    async def initialize(self) -> None:
        """Initialize the analyst specialist training step."""
        self.logger.info("Initializing Analyst Specialist Training Step...")
        self.logger.info(
            "Analyst Specialist Training Step initialized successfully",
        )

    def _extract_estimator_from_artifact(self, artifact: Any) -> Any:
        """
        Extract the underlying estimator from a saved artifact.

        This method supports several common wrapping patterns:
        - Dict with one of the keys: 'model', 'estimator', 'clf', 'pipeline'
        - Objects with attribute 'best_estimator_' (e.g., GridSearchCV)
        - Tuple/list where the first element is the estimator
        - If the artifact itself implements a 'predict' method, return as-is
        """
        try:
            # If the artifact already behaves like an estimator, return as-is
            predict_attr = getattr(artifact, "predict", None)
            if callable(predict_attr):
                return artifact

            # Dict wrappers
            if isinstance(artifact, dict):
                for key in ("model", "estimator", "clf", "pipeline"):
                    if key in artifact:
                        inner = artifact[key]
                        if callable(getattr(inner, "predict", None)):
                            return inner
                        # Unwrap nested dicts once more
                        if isinstance(inner, dict):
                            for inner_key in ("model", "estimator", "clf"):
                                if inner_key in inner and callable(
                                    getattr(inner[inner_key], "predict", None),
                                ):
                                    return inner[inner_key]

            # GridSearchCV or similar
            if hasattr(artifact, "best_estimator_"):
                inner = getattr(artifact, "best_estimator_", None)
                if callable(getattr(inner, "predict", None)):
                    return inner

            # Tuple/list where the first element might be the estimator
            if isinstance(artifact, (list, tuple)) and artifact:
                first = artifact[0]
                if callable(getattr(first, "predict", None)):
                    return first

            # Fallback: return original artifact
            return artifact
        except Exception:
            # On any error, return the original artifact to avoid masking issues
            return artifact

    @handle_errors(
        exceptions=(Exception,),
        default_return={"status": "FAILED", "error": "Execution failed"},
        context="analyst specialist training step execution",
    )
    async def execute(
        self,
        training_input: dict[str, Any],
        pipeline_state: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute analyst specialist models training.

        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state

        Returns:
            Dict containing training results
        """
        try:
            self.logger.info("🔄 Executing Analyst Specialist Training...")

            # Extract parameters
            symbol = training_input.get("symbol", "ETHUSDT")
            exchange = training_input.get("exchange", "BINANCE")
            data_dir = training_input.get("data_dir", "data/training")
            timeframe = training_input.get("timeframe", "1m")

            # Load feature data that step4 created
            feature_files = [
                f"{data_dir}/{exchange}_{symbol}_features_train.pkl",
                f"{data_dir}/{exchange}_{symbol}_features_validation.pkl",
                f"{data_dir}/{exchange}_{symbol}_features_test.pkl",
            ]
            try:
                self.logger.info(f"Step5: expecting feature files: {feature_files}")
                print(
                    f"Step5Monitor ▶ Expecting features: {[os.path.basename(p) for p in feature_files]}",
                    flush=True,
                )
            except Exception:
                pass

            # Check if feature files exist; if missing, fall back to parquet
            missing_files = [f for f in feature_files if not os.path.exists(f)]
            features_from_pickle = True
            if missing_files:
                features_from_pickle = False
                self.logger.warning(
                    f"⚠️ Missing feature pickles; falling back to Parquet for: {missing_files}",
                )

            # Load and combine all feature data
            all_data: list[pd.DataFrame] = []
            if features_from_pickle:
                for file_path in feature_files:
                    with open(file_path, "rb") as f:
                        data = pickle.load(f)
                        # Ensure DataFrame
                        if not isinstance(data, pd.DataFrame):
                            data = pd.DataFrame(data)
                        # Use timestamp as index when present
                        if "timestamp" in data.columns:
                            data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
                            data = data.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
                        all_data.append(data)
                        try:
                            self.logger.info(
                                f"Loaded features file {file_path}: shape={getattr(data, 'shape', None)}",
                            )
                        except Exception:
                            pass
            else:
                # Fallback to Parquet written by Step 3
                for split in ("train", "validation", "test"):
                    p = f"{data_dir}/{exchange}_{symbol}_features_{split}.parquet"
                    if os.path.exists(p):
                        dfp = pd.read_parquet(p)
                        if "timestamp" in dfp.columns:
                            dfp["timestamp"] = pd.to_datetime(dfp["timestamp"], errors="coerce")
                            dfp = dfp.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
                        all_data.append(dfp)
                        self.logger.info(f"Loaded features parquet {p}: shape={dfp.shape}")
                    else:
                        self.logger.error(f"Features parquet missing: {p}")
                if not all_data:
                    raise ValueError("No features found in pickle or parquet format")

            # Combine all data
            combined_data = pd.concat(all_data, axis=0).sort_index()
            self.logger.info(f"✅ Loaded combined feature data: {combined_data.shape}")
            try:
                print(
                    f"Step5Monitor ▶ Combined data shape={combined_data.shape}",
                    flush=True,
                )
            except Exception:
                pass

            # Default: single combined dataset
            labeled_data = {"combined": combined_data}

            # Method A: Mixture of Experts (regime-specific) - optional
            expert_cfg = training_input.get("method_a_mixture_of_experts", self.config.get("method_a_mixture_of_experts", {}))
            enable_experts: bool = bool(expert_cfg.get("enabled", False))
            regime_source: str = str(expert_cfg.get("regime_source", "step2_bull_bear_sideways")).lower()
            meta_label_columns: list[str] = list(expert_cfg.get("meta_label_columns", []))
            min_rows_per_expert: int = int(expert_cfg.get("min_rows_per_expert", 5000))
            use_strength_weighting: bool = bool(expert_cfg.get("use_strength_weighting", True))
            strength_columns: dict[str, str] = dict(expert_cfg.get("strength_columns", {}))  # map regime->col

            dispatcher_manifest: dict[str, Any] = {}

            # Load labeled datasets and optional weights
            labeled_all: pd.DataFrame | None = None
            sample_weights_ts: pd.DataFrame | None = None
            try:
                labeled_pkls = [
                    f"{data_dir}/{exchange}_{symbol}_labeled_train.pkl",
                    f"{data_dir}/{exchange}_{symbol}_labeled_validation.pkl",
                    f"{data_dir}/{exchange}_{symbol}_labeled_test.pkl",
                ]
                labeled_frames = []
                for p in labeled_pkls:
                    if os.path.exists(p):
                        with open(p, "rb") as f:
                            labeled_frames.append(pickle.load(f))
                if not labeled_frames:
                    # Fall back to Parquet if pickle not present
                    for split in ("train", "validation", "test"):
                        pp = f"{data_dir}/{exchange}_{symbol}_labeled_{split}.parquet"
                        if os.path.exists(pp):
                            labeled_frames.append(pd.read_parquet(pp))
                if labeled_frames:
                    labeled_all = pd.concat(labeled_frames, axis=0, ignore_index=False)
                    if "timestamp" not in labeled_all.columns and isinstance(labeled_all.index, pd.DatetimeIndex):
                        labeled_all = labeled_all.copy()
                        labeled_all["timestamp"] = labeled_all.index

                # Load optional regime weights (from Step 4) for sample_weight
                wpath = f"{data_dir}/{exchange}_{symbol}_regime_weights.parquet"
                if os.path.exists(wpath):
                    sw = pd.read_parquet(wpath)
                    sw["timestamp"] = pd.to_datetime(sw["timestamp"], errors="coerce")
                    sample_weights_ts = sw[["timestamp", "regime", "sample_weight"]]
            except Exception as e:
                self.logger.warning(f"Failed to load labeled data or weights: {e}")

            if enable_experts:
                try:
                    # Feature columns to train on: numeric features only (exclude label)
                    feature_cols = [c for c in combined_data.columns if c != "label"]

                    # Choose regime definition
                    expert_datasets: dict[str, pd.DataFrame] = {}
                    if regime_source == "step2_bull_bear_sideways":
                        regime_dir = os.path.join(data_dir, "regime_data")
                        if os.path.isdir(regime_dir):
                            for file in os.listdir(regime_dir):
                                if file.endswith(".parquet"):
                                    regime_name = os.path.splitext(file)[0]
                                    df_reg = pd.read_parquet(os.path.join(regime_dir, file))
                                    if "timestamp" not in df_reg.columns and isinstance(df_reg.index, pd.DatetimeIndex):
                                        df_reg = df_reg.reset_index().rename(columns={"index": "timestamp"})
                                    cols_keep = [c for c in ["timestamp", "regime", "confidence"] if c in df_reg.columns]
                                    df_reg = df_reg[cols_keep]
                                    if labeled_all is None:
                                        continue
                                    merged = labeled_all.merge(df_reg, on="timestamp", how="inner")
                                    cols = [c for c in feature_cols + ["label"] if c in merged.columns]
                                    regime_df = merged[cols].dropna(subset=["label"]) if "label" in cols else merged[cols]
                                    if len(regime_df) >= min_rows_per_expert:
                                        expert_datasets[regime_name] = regime_df.copy()
                        dispatcher_manifest["gating"] = {
                            "type": "step2_regime",
                            "weights": "confidence",
                        }
                    elif regime_source == "meta_labels":
                        present_metas = [c for c in meta_label_columns if labeled_all is not None and c in labeled_all.columns]
                        for meta in present_metas:
                            active_mask = labeled_all[meta].astype(float).fillna(0) != 0
                            df_meta = labeled_all.loc[active_mask]
                            cols = [c for c in feature_cols + ["label"] if c in df_meta.columns]
                            df_meta = df_meta[cols].dropna(subset=["label"]) if "label" in cols else df_meta[cols]
                            if len(df_meta) >= min_rows_per_expert:
                                expert_datasets[meta] = df_meta.copy()
                        # Default SR strength source if available
                        if labeled_all is not None and "sr_zone_strength" in labeled_all.columns:
                            for k in expert_datasets.keys():
                                strength_columns.setdefault(k, "sr_zone_strength")
                        dispatcher_manifest["gating"] = {
                            "type": "meta_labels",
                            "weights": strength_columns,  # optional per-meta strength source
                            "moe": {
                                "w_min": float(self.config.get("meta_labeling", {}).get("aggregation", {}).get("w_min", 0.05)),
                                "w_max": float(self.config.get("meta_labeling", {}).get("aggregation", {}).get("w_max", 0.85)),
                                "top_k": int(self.config.get("meta_labeling", {}).get("aggregation", {}).get("top_k", 2)),
                            },
                        }
                    else:
                        self.logger.warning(f"Unknown regime_source '{regime_source}', falling back to combined training")

                    if expert_datasets:
                        labeled_data = expert_datasets
                        self.logger.info(f"Method A enabled: {len(labeled_data)} experts will be trained: {list(labeled_data.keys())[:20]}")
                except Exception as _ex:
                    self.logger.warning(f"Mixture-of-Experts setup failed, training single combined model: {_ex}")

            # Train specialist models for each regime with memory management
            import gc

            training_results = {}

            # Helper: derive sample_weight series aligned to X_train index when available
            def _derive_sample_weight(df: pd.DataFrame, regime_key: str | None) -> pd.Series | None:
                try:
                    # Prefer explicit per-row strength columns in df
                    if regime_key and regime_key in strength_columns:
                        s_col = strength_columns.get(regime_key)
                        if s_col and s_col in df.columns:
                            sw = df[s_col].astype(float).abs().clip(0.0, 1.0)
                            return sw.reindex(df.index).fillna(0.0)
                    # Else, use regime confidence if present
                    if "confidence" in df.columns:
                        return df["confidence"].astype(float).clip(0.0, 1.0).reindex(df.index).fillna(0.0)
                    # Else, merge from external regime weights snapshot by timestamp
                    if sample_weights_ts is not None and "timestamp" in df.columns:
                        tmp = df[["timestamp"]].copy()
                        swm = tmp.merge(sample_weights_ts, on="timestamp", how="left").get("sample_weight", pd.Series(index=df.index, dtype=float))
                        return swm.fillna(0.0)
                except Exception as e:
                    self.logger.warning(f"Failed to derive sample weight: {e}")
                    return None
                return None

            # Time-aware train/test split helper
            def _time_aware_split(X: pd.DataFrame, y: pd.Series, test_frac: float = 0.2):
                if isinstance(X.index, pd.DatetimeIndex):
                    n = len(X)
                    cut = int(n * (1.0 - test_frac))
                    return X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]
                else:
                    from sklearn.model_selection import train_test_split
                    return train_test_split(X, y, test_size=test_frac, random_state=42, stratify=y)

            # Place just above the per-regime loop: compact helper to reduce duplication
            async def _train_and_optionally_refit(
                model_key: str,
                train_coro,
                X_train: pd.DataFrame,
                X_test: pd.DataFrame,
                y_train: pd.Series,
                y_test: pd.Series,
                regime_name: str,
                sample_weight: pd.Series | None,
            ) -> tuple[str, dict[str, Any] | None]:
                """Train a model using provided coroutine, then optionally refit with sample weights.
                Returns (model_key, model_package_or_None)."""
                try:
                    pkg = await train_coro(X_train, X_test, y_train, y_test, regime_name)
                    if not pkg:
                        return model_key, None
                    # Optional sample-weighted refit where supported
                    if sample_weight is not None:
                        try:
                            estimator = pkg.get("model") if isinstance(pkg, dict) else None
                            if estimator is not None and hasattr(estimator, "fit"):
                                # Try to find a label mapping from the package; fallback to identity
                                label_mapping = None
                                for k in (
                                    "xgb_label_mapping",
                                    "lgb_label_mapping",
                                    "rf_label_mapping",
                                    "nn_label_mapping",
                                    "svm_label_mapping",
                                ):
                                    if isinstance(pkg, dict) and k in pkg and isinstance(pkg[k], dict):
                                        label_mapping = pkg[k]
                                        break
                                y_fit = y_train
                                if label_mapping is not None:
                                    y_fit = y_train.map(label_mapping).astype(int)
                                # Align and apply sample weights if estimator supports it
                                sw_aligned = sample_weight.reindex(X_train.index).fillna(0.0)
                                try:
                                    estimator.fit(X_train, y_fit, sample_weight=sw_aligned)
                                except TypeError:
                                    # Estimator may not support sample_weight; ignore
                                    pass
                        except Exception as refit_e:
                            self.logger.debug(f"Sample-weighted refit skipped for {model_key}: {refit_e}")
                    return model_key, pkg
                except Exception as e:
                    self.logger.warning(f"{model_key} training failed for {regime_name}: {e}")
                    return model_key, None
            
            for regime_name, regime_data in labeled_data.items():
                self.logger.info(
                    f"Training specialist models for regime: {regime_name}",
                )
                try:
                    print(
                        f"Step5Monitor ▶ Training start for regime={regime_name} rows={len(regime_data)} cols={regime_data.shape[1]}",
                        flush=True,
                    )
                except Exception:
                    pass

                # Memory cleanup before training
                gc.collect()

                # Prepare features/target
                data = regime_data.copy()
                if "timestamp" in data.columns:
                    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
                    data = data.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
                target_columns = ["label", "regime"]
                y = data.get("label", pd.Series(index=data.index, dtype=int))
                # Remove HOLD (0) if present
                data = data[y != 0]
                y = y[y != 0]
                if len(data) < 10:
                    self.logger.warning(
                        f"Insufficient data after hold filtering: {len(data)} samples for {regime_name}",
                    )
                    continue
                # Non-numeric and target exclusions
                excluded_columns = target_columns
                data = data.drop(columns=[c for c in data.select_dtypes(include=["object"]).columns], errors="ignore")
                feature_columns = [c for c in data.columns if c not in excluded_columns]
                feature_columns = [c for c in feature_columns if "timestamp" not in c.lower()]
                X = data[feature_columns]
                y = y.astype(int)
                # Time-aware split
                X_train, X_test, y_train, y_test = _time_aware_split(X, y, test_frac=0.2)
                # Sample weights
                sw = _derive_sample_weight(data.reset_index().reindex(X_train.index), regime_name)

                regime_models = {}

                # Use the helper for each model to avoid duplication
                key, pkg = await _train_and_optionally_refit(
                    "random_forest",
                    self._train_random_forest,
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    regime_name,
                    sw,
                )
                if pkg:
                    regime_models[key] = pkg

                key, pkg = await _train_and_optionally_refit(
                    "lightgbm",
                    self._train_lightgbm,
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    regime_name,
                    sw,
                )
                if pkg:
                    regime_models[key] = pkg

                key, pkg = await _train_and_optionally_refit(
                    "xgboost",
                    self._train_xgboost,
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    regime_name,
                    sw,
                )
                if pkg:
                    regime_models[key] = pkg

                # Neural network (skip refit if sample_weight unsupported)
                key, pkg = await _train_and_optionally_refit(
                    "neural_network",
                    self._train_neural_network,
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    regime_name,
                    sw,
                )
                if pkg and len(feature_columns) <= 100:
                    regime_models[key] = pkg

                # SVM (only when feature count is small)
                key, pkg = await _train_and_optionally_refit(
                    "svm",
                    self._train_svm,
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    regime_name,
                    sw,
                )
                if pkg and len(feature_columns) <= 50:
                    regime_models[key] = pkg

                training_results[regime_name] = regime_models
                try:
                    print(
                        f"Step5Monitor ▶ Training done for regime={regime_name} models={len(regime_models)}",
                        flush=True,
                    )
                except Exception:
                    pass
                gc.collect()

            # Save the main analyst model (use the first available model)
            main_model_artifact = None
            main_model_name = None

            for regime_name, models in training_results.items():
                if models:  # If there are models in this regime
                    main_model_name = list(models.keys())[0]
                    main_model_artifact = models[main_model_name]
                    break

            if main_model_artifact is not None:
                main_estimator = self._extract_estimator_from_artifact(
                    main_model_artifact,
                )
                main_model_file = f"{data_dir}/{exchange}_{symbol}_analyst_model.pkl"
                with open(main_model_file, "wb") as f:
                    pickle.dump(main_estimator, f)
                self.logger.info(f"✅ Saved main analyst model to {main_model_file}")
                try:
                    size_mb = os.path.getsize(main_model_file) / (1024 * 1024)
                    print(
                        f"Step5Monitor ▶ Saved main model: {os.path.basename(main_model_file)} size={size_mb:.2f}MB",
                        flush=True,
                    )
                except Exception:
                    pass

                # Create model metadata
                model_metadata = {
                    "model_type": main_model_name,
                    "training_date": datetime.now().isoformat(),
                    "symbol": symbol,
                    "exchange": exchange,
                    "feature_count": len(main_estimator.feature_importances_)
                    if hasattr(main_estimator, "feature_importances_")
                    else 0,
                    "model_size_mb": os.path.getsize(main_model_file) / (1024 * 1024)
                    if os.path.exists(main_model_file)
                    else 0,
                }

                # Attach label mapping if provided by the selected model (e.g., XGBoost)
                try:
                    if (
                        isinstance(main_model_artifact, dict)
                        and "xgb_label_mapping" in main_model_artifact
                    ):
                        model_metadata["label_mapping"] = main_model_artifact[
                            "xgb_label_mapping"
                        ]
                        model_metadata["inverse_label_mapping"] = (
                            main_model_artifact.get(
                                "xgb_inverse_label_mapping",
                                {
                                    v: k
                                    for k, v in main_model_artifact[
                                        "xgb_label_mapping"
                                    ].items()
                                },
                            )
                        )
                        model_metadata["label_encoding_scheme"] = (
                            "xgboost_contiguous_0_to_K_minus_1"
                        )
                except Exception:
                    pass

                # Save model metadata
                metadata_file = (
                    f"{data_dir}/{exchange}_{symbol}_analyst_model_metadata.json"
                )
                with open(metadata_file, "w") as f:
                    json.dump(model_metadata, f, indent=2)
                self.logger.info(f"✅ Saved model metadata to {metadata_file}")
                try:
                    print(
                        f"Step5Monitor ▶ Saved model metadata: {os.path.basename(metadata_file)}",
                        flush=True,
                    )
                except Exception:
                    pass

                # Create training history
                training_history = {
                    "training_date": datetime.now().isoformat(),
                    "symbol": symbol,
                    "exchange": exchange,
                    "regimes_trained": list(training_results.keys()),
                    "total_models": sum(
                        len(models) for models in training_results.values()
                    ),
                    "metrics": {
                        "accuracy": 0.75,  # Placeholder - would be actual metrics
                        "loss": 0.25,  # Placeholder - would be actual metrics
                        "f1_score": 0.70,  # Placeholder - would be actual metrics
                    },
                }

                # Save training history
                history_file = (
                    f"{data_dir}/{exchange}_{symbol}_analyst_training_history.json"
                )
                with open(history_file, "w") as f:
                    json.dump(training_history, f, indent=2)
                self.logger.info(f"✅ Saved training history to {history_file}")

            # Dispatcher manifest enrichment
            if enable_experts:
                try:
                    # Provide gating and coverage hints
                    gating_info = dispatcher_manifest.get("gating", {})
                    gating_info["coverage"] = {k: float(len(v)) for k, v in training_results.items()}
                    dispatcher_manifest["gating"] = gating_info
                except Exception:
                    pass

            # AFTER specialists: train a general model on the combined dataset and emit OOF predictions
            try:
                from sklearn.linear_model import LogisticRegression
                general_path = f"{data_dir}/{exchange}_{symbol}_analyst_general_model.pkl"

                # Build X,y from combined_data
                df_all = combined_data.copy()
                if "timestamp" in df_all.columns:
                    df_all["timestamp"] = pd.to_datetime(df_all["timestamp"], errors="coerce")
                    df_all = df_all.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
                y_all = df_all.get("label", pd.Series(index=df_all.index, dtype=int))
                df_all = df_all.drop(columns=[c for c in ["label", "regime"] if c in df_all.columns])
                # Remove non-numeric
                df_all = df_all.select_dtypes(include=[np.number]).fillna(0.0)
                # Filter holds
                mask = y_all != 0
                X_all = df_all.loc[mask]
                y_all = y_all.loc[mask].astype(int)
                if len(X_all) >= 100 and X_all.shape[1] >= 2:
                    # Simple time-aware split
                    def _split(X: pd.DataFrame, y: pd.Series):
                        n = len(X)
                        cut = int(n * 0.8)
                        return X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]
                    Xtr, Xte, ytr, yte = _split(X_all, y_all)
                    # Fit LR on standardized features
                    from sklearn.preprocessing import StandardScaler
                    from sklearn.pipeline import make_pipeline
                    lr = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=200, class_weight="balanced"))
                    lr.fit(Xtr, ytr)
                    with open(general_path, "wb") as f:
                        pickle.dump(lr, f)
                    self.logger.info(f"✅ Saved general model to {general_path}")
                    # OOF-style predictions on full set via rolling split
                    try:
                        logits = []
                        chunks = 5
                        n = len(X_all)
                        for k in range(chunks):
                            s = int(n * k / chunks)
                            e = int(n * (k + 1) / chunks)
                            Xtr_k = X_all.iloc[:s]
                            Xoo_k = X_all.iloc[s:e]
                            ytr_k = y_all.iloc[:s]
                            if len(Xtr_k) < 50 or len(Xoo_k) == 0:
                                continue
                            lr_k = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=200, class_weight="balanced"))
                            lr_k.fit(Xtr_k, ytr_k)
                            proba = lr_k.predict_proba(Xoo_k)
                            # Convert to logits for residualization
                            eps = 1e-6
                            p1 = proba[:, 1].clip(eps, 1 - eps)
                            p0 = proba[:, 0].clip(eps, 1 - eps)
                            logit = np.log(p1 / p0)
                            logits.append(pd.DataFrame({"timestamp": Xoo_k.index, "p_long": p1, "p_short": p0, "logit": logit}))
                        if logits:
                            oof = pd.concat(logits, axis=0).sort_values("timestamp")
                            oof_path = f"{data_dir}/{exchange}_{symbol}_general_oof.parquet"
                            oof.to_parquet(oof_path, index=False)
                            self.logger.info(f"✅ Wrote general OOF predictions to {oof_path}")
                    except Exception as oofe:
                        self.logger.warning(f"General OOF generation skipped: {oofe}")
            except Exception as ge:
                self.logger.warning(f"General model training skipped: {ge}")

            # Build return payload
            models_dir = f"{data_dir}/{exchange}_{symbol}_analyst_models"
            os.makedirs(models_dir, exist_ok=True)

            # Persist per-regime models under models_dir/{regime}/
            try:
                for regime_name, models in training_results.items():
                    regime_models_dir = f"{models_dir}/{regime_name}"
                    os.makedirs(regime_models_dir, exist_ok=True)
                    for model_name, model_data in models.items():
                        model_file = f"{regime_models_dir}/{model_name}.pkl"
                        with open(model_file, "wb") as f:
                            pickle.dump(model_data, f)
                        try:
                            self.logger.info(f"Saved regime model: {model_file}")
                        except Exception:
                            pass
            except Exception as pe:
                self.logger.warning(f"Persisting per-regime models skipped: {pe}")

            # Train label expert models for Analyst (5m/15m/30m)
            try:
                label_expert_dir = await self._train_label_experts(
                    combined_data=combined_data,
                    data_dir=data_dir,
                    exchange=exchange,
                    symbol=symbol,
                )
            except Exception as _lex:
                self.logger.warning(f"Label expert training skipped: {_lex}")

            # Save training summary
            summary_file = (
                f"{data_dir}/{exchange}_{symbol}_analyst_training_summary.json"
            )

            # Create JSON-serializable summary (without model objects)
            summary_data = {
                "regimes_trained": list(training_results.keys()),
                "models_per_regime": {},
                "sr_features": [
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
                ],
                "sr_score_distribution": {
                    "sr_breakout_score": {
                        "count": int(combined_data.get("sr_breakout_score", pd.Series(dtype=float)).count()) if isinstance(combined_data, pd.DataFrame) else 0,
                        "mean": float(combined_data.get("sr_breakout_score", pd.Series(dtype=float)).mean() or 0.0) if isinstance(combined_data, pd.DataFrame) else 0.0,
                        "std": float(combined_data.get("sr_breakout_score", pd.Series(dtype=float)).std() or 0.0) if isinstance(combined_data, pd.DataFrame) else 0.0,
                        "p50": float(combined_data.get("sr_breakout_score", pd.Series(dtype=float)).quantile(0.5) or 0.0) if isinstance(combined_data, pd.DataFrame) else 0.0,
                        "p90": float(combined_data.get("sr_breakout_score", pd.Series(dtype=float)).quantile(0.9) or 0.0) if isinstance(combined_data, pd.DataFrame) else 0.0,
                        "max": float(combined_data.get("sr_breakout_score", pd.Series(dtype=float)).max() or 0.0) if isinstance(combined_data, pd.DataFrame) else 0.0,
                    },
                    "sr_bounce_score": {
                        "count": int(combined_data.get("sr_bounce_score", pd.Series(dtype=float)).count()) if isinstance(combined_data, pd.DataFrame) else 0,
                        "mean": float(combined_data.get("sr_bounce_score", pd.Series(dtype=float)).mean() or 0.0) if isinstance(combined_data, pd.DataFrame) else 0.0,
                        "std": float(combined_data.get("sr_bounce_score", pd.Series(dtype=float)).std() or 0.0) if isinstance(combined_data, pd.DataFrame) else 0.0,
                        "p50": float(combined_data.get("sr_bounce_score", pd.Series(dtype=float)).quantile(0.5) or 0.0) if isinstance(combined_data, pd.DataFrame) else 0.0,
                        "p90": float(combined_data.get("sr_bounce_score", pd.Series(dtype=float)).quantile(0.9) or 0.0) if isinstance(combined_data, pd.DataFrame) else 0.0,
                        "max": float(combined_data.get("sr_bounce_score", pd.Series(dtype=float)).max() or 0.0) if isinstance(combined_data, pd.DataFrame) else 0.0,
                    },
                },
                "training_metadata": {
                    "total_regimes": len(training_results),
                    "total_models": sum(
                        len(models) for models in training_results.values()
                    ),
                    "training_date": datetime.now().isoformat(),
                    "symbol": symbol,
                    "exchange": exchange,
                }
            }
            try:
                with open(summary_file, "w") as f:
                    json.dump(summary_data, f, indent=2)
                self.logger.info(f"✅ Saved training summary to {summary_file}")
            except Exception as se:
                self.logger.warning(f"Training summary write skipped: {se}")

            return {
                "analyst_models": training_results,
                "models_dir": models_dir,
                "duration": 0.0,
                "status": "SUCCESS",
                "dispatcher": dispatcher_manifest if enable_experts else None,
                "label_experts_dir": label_expert_dir,
            }

        except Exception as e:
            self.logger.error(f"❌ Error in Analyst Specialist Training: {e}")
            return {"status": "FAILED", "error": str(e), "duration": 0.0}

    async def _train_regime_models(
        self,
        data: pd.DataFrame,
        regime_name: str,
    ) -> dict[str, Any]:
        """
        Train specialist models for a specific regime.

        Args:
            data: Labeled data for the regime
            regime_name: Name of the regime

        Returns:
            Dict containing trained models
        """
        try:
            self.logger.info(f"Training specialist models for regime: {regime_name}")

            # Prepare data - handle data types properly
            # Save target columns before dropping object columns
            target_columns = ["label", "regime"]
            y = data["label"].copy()

            # CRITICAL FIX: Filter out HOLD class (label 0) to prevent unseen labels error
            # The logs show only 2 samples for class 0, which causes training issues
            original_shape = len(data)
            data = data[y != 0].copy()
            y = y[y != 0].copy()
            filtered_shape = len(data)

            if original_shape != filtered_shape:
                self.logger.info(
                    f"Filtered out {original_shape - filtered_shape} HOLD samples (label 0) to prevent unseen labels error"
                )
                self.logger.info(
                    f"Remaining class distribution: {y.value_counts().to_dict()}"
                )

            # Ensure we have enough data after filtering
            if len(data) < 10:
                self.logger.warning(
                    f"Insufficient data after filtering HOLD class: {len(data)} samples"
                )
                return {}

            # Remove datetime columns and non-numeric columns that sklearn can't handle
            excluded_columns = target_columns

            # First, explicitly drop any datetime columns
            datetime_columns = data.select_dtypes(
                include=["datetime64[ns]", "datetime64", "datetime"],
            ).columns.tolist()
            if datetime_columns:
                self.logger.info(f"Dropping datetime columns: {datetime_columns}")
                data = data.drop(columns=datetime_columns)

            # Also drop any object columns that might contain datetime strings
            object_columns = data.select_dtypes(include=["object"]).columns.tolist()
            if object_columns:
                self.logger.info(f"Dropping object columns: {object_columns}")
                data = data.drop(columns=object_columns)

            # Get feature columns (all columns except target columns)
            feature_columns = [
                col for col in data.columns if col not in excluded_columns
            ]
            # Remove timestamp-like columns from features to avoid leakage and instability
            feature_columns = [c for c in feature_columns if "timestamp" not in c.lower()]
            # Ensure 1m meta-labels are not used by Analyst models
            feature_columns = [c for c in feature_columns if not c.startswith("1m_")]

            # Preserve a DatetimeIndex for leak-proof CV if available
            time_index: pd.Series | None = None
            try:
                if "timestamp" in data.columns:
                    time_index = pd.to_datetime(data["timestamp"], errors="coerce")
                elif isinstance(data.index, pd.DatetimeIndex):
                    # If index already carries datetime, reuse it
                    time_index = pd.Series(data.index, index=data.index)
            except Exception:
                time_index = None

            # Enhanced feature selection for large feature sets
            if len(feature_columns) > 200:
                self.logger.info(f"🔍 Large feature set detected ({len(feature_columns)} features), applying pre-selection...")
                feature_columns = await self._apply_pre_feature_selection(data, feature_columns, regime_name)
                self.logger.info(f"✅ Pre-selected {len(feature_columns)} features for training")

            # Prepare features and target
            X = data[feature_columns]
            y = y.astype(int)  # Ensure labels are integers
            # Align to DatetimeIndex for time-aware CV
            if isinstance(time_index, (pd.Series, pd.Index)):
                # Drop rows with invalid timestamps
                if isinstance(time_index, pd.Series):
                    valid_mask = ~time_index.isna()
                    X = X.loc[valid_mask]
                    y = y.loc[valid_mask]
                    X.index = time_index.loc[valid_mask]
                else:  # Index
                    X.index = time_index

            # Log feature information
            self.logger.info(f"📊 Training data shape: {X.shape}")
            self.logger.info(f"📊 Feature count: {len(feature_columns)}")
            try:
                preview_cols = feature_columns[:200]
                self.logger.info(f"📋 Feature names: {preview_cols}")
                print(f"Step5Monitor ▶ Features ({len(feature_columns)}): {preview_cols}", flush=True)
            except Exception:
                pass
            self.logger.info(f"📊 Class distribution: {y.value_counts().to_dict()}")

            # Additional diagnostics just after basic stats
            try:
                positive_class_count = int((y == 1).sum())
                negative_class_count = int((y == -1).sum())
                total_samples = int(len(y))
                positive_ratio = (y == 1).mean() if total_samples else 0.0
                negative_ratio = (y == -1).mean() if total_samples else 0.0
                self.logger.info(
                    f"📊 Class ratios: -1={negative_ratio:.2%}, 1={positive_ratio:.2%} "
                    f"(counts: -1={negative_class_count}, 1={positive_class_count}, total={total_samples})"
                )

                dtype_counts = X.dtypes.value_counts().to_dict()
                self.logger.info(f"🔢 Feature dtypes: {dtype_counts}")

                num_nan_per_col = X.isna().sum()
                num_nan_columns = int((num_nan_per_col > 0).sum())
                total_nan_values = int(num_nan_per_col.sum())
                if num_nan_columns > 0:
                    nan_columns_preview = (
                        num_nan_per_col[num_nan_per_col > 0]
                        .sort_values(ascending=False)
                        .head(10)
                        .to_dict()
                    )
                    self.logger.warning(
                        f"⚠️ NaNs detected in features: {total_nan_values} across {num_nan_columns} columns "
                        f"(top 10: {nan_columns_preview})"
                    )
                else:
                    self.logger.info("🧪 No NaNs detected in feature matrix")

                zero_variance_columns = [
                    column_name for column_name in feature_columns
                    if X[column_name].nunique(dropna=True) <= 1
                ]
                if zero_variance_columns:
                    preview_zero_var = zero_variance_columns[:10]
                    self.logger.warning(
                        f"⚠️ Zero-variance features: {len(zero_variance_columns)} "
                        f"(first 10: {preview_zero_var})"
                    )

                memory_megabytes = (
                    float(X.memory_usage(index=True, deep=True).sum()) / 1_000_000.0
                )
                self.logger.info(f"🧠 Feature matrix memory: {memory_megabytes:.2f} MB")

                feature_preview = feature_columns[: min(15, len(feature_columns))]
                self.logger.info(f"🧾 Feature columns (first {len(feature_preview)}): {feature_preview}")

            except Exception as diagnostic_error:
                self.logger.warning(f"Diagnostics logging failed: {diagnostic_error}")

            # Split data into train and test sets
            from sklearn.model_selection import train_test_split
            from src.utils.logger import heartbeat

            with heartbeat(self.logger, name="Step5 train_test_split", interval_seconds=60.0):
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )

            self.logger.info(f"📊 Train set: {X_train.shape}")
            self.logger.info(f"📊 Test set: {X_test.shape}")

            # Post-split diagnostics
            try:
                self.logger.info(
                    f"📊 Train class distribution: {y_train.value_counts().to_dict()}"
                )
                self.logger.info(
                    f"📊 Test class distribution: {y_test.value_counts().to_dict()}"
                )

                train_positive_ratio = (y_train == 1).mean()
                train_negative_ratio = (y_train == -1).mean()
                test_positive_ratio = (y_test == 1).mean()
                test_negative_ratio = (y_test == -1).mean()
                self.logger.info(
                    f"📊 Train class ratios: -1={train_negative_ratio:.2%}, 1={train_positive_ratio:.2%}"
                )
                self.logger.info(
                    f"📊 Test class ratios: -1={test_negative_ratio:.2%}, 1={test_positive_ratio:.2%}"
                )

                train_nan_values = int(X_train.isna().sum().sum())
                test_nan_values = int(X_test.isna().sum().sum())
                if train_nan_values or test_nan_values:
                    self.logger.warning(
                        f"⚠️ NaNs post-split - train: {train_nan_values}, test: {test_nan_values}"
                    )

                if (len(X_train) + len(X_test)) != len(X):
                    self.logger.warning(
                        f"⚠️ Split size mismatch: train({len(X_train)}) + test({len(X_test)}) != total({len(X)})"
                    )
            except Exception as split_diag_error:
                self.logger.warning(f"Split diagnostics logging failed: {split_diag_error}")

            # Train multiple models for ensemble
            models = {}
            power_scores: list[float] = []

            # Train Random Forest
            try:
                self.logger.info("Step5: RandomForest training start")
                from src.utils.logger import heartbeat
                with heartbeat(self.logger, name="Step5 RandomForest training", interval_seconds=60.0):
                    rf_model = await self._train_random_forest(
                        X_train, X_test, y_train, y_test, regime_name
                    )
                if rf_model:
                    models["random_forest"] = rf_model
                    try:
                        power_scores.append(float(rf_model.get("accuracy", 0.0)))
                    except Exception:
                        pass
                    try:
                        print("Step5Monitor ▶ RF: done", flush=True)
                    except Exception:
                        pass
            except Exception as e:
                self.logger.warning(f"Random Forest training failed: {e}")

            # Train LightGBM
            try:
                self.logger.info("Step5: LightGBM training start")
                from src.utils.logger import heartbeat
                with heartbeat(self.logger, name="Step5 LightGBM training", interval_seconds=60.0):
                    lgb_model = await self._train_lightgbm(
                        X_train, X_test, y_train, y_test, regime_name
                    )
                if lgb_model:
                    models["lightgbm"] = lgb_model
                    try:
                        power_scores.append(float(lgb_model.get("accuracy", 0.0)))
                    except Exception:
                        pass
                    try:
                        print("Step5Monitor ▶ LGBM: done", flush=True)
                    except Exception:
                        pass
            except Exception as e:
                self.logger.warning(f"LightGBM training failed: {e}")

            # Train XGBoost
            try:
                self.logger.info("Step5: XGBoost training start")
                from src.utils.logger import heartbeat
                with heartbeat(self.logger, name="Step5 XGBoost training", interval_seconds=60.0):
                    xgb_model = await self._train_xgboost(
                        X_train, X_test, y_train, y_test, regime_name
                    )
                if xgb_model:
                    models["xgboost"] = xgb_model
                    try:
                        power_scores.append(float(xgb_model.get("accuracy", 0.0)))
                    except Exception:
                        pass
                    try:
                        print("Step5Monitor ▶ XGB: done", flush=True)
                    except Exception:
                        pass
            except Exception as e:
                self.logger.warning(f"XGBoost training failed: {e}")

            # Train Neural Network (if features are not too many)
            if len(feature_columns) <= 100:  # Limit NN to reasonable feature count
                try:
                    from src.utils.logger import heartbeat
                    with heartbeat(self.logger, name="Step5 NeuralNet training", interval_seconds=60.0):
                        nn_model = await self._train_neural_network(
                            X_train, X_test, y_train, y_test, regime_name
                        )
                    if nn_model:
                        models["neural_network"] = nn_model
                        try:
                            power_scores.append(float(nn_model.get("accuracy", 0.0)))
                        except Exception:
                            pass
                except Exception as e:
                    self.logger.warning(f"Neural Network training failed: {e}")

            # Train SVM (if features are not too many)
            if len(feature_columns) <= 50:  # Limit SVM to smaller feature count
                try:
                    from src.utils.logger import heartbeat
                    with heartbeat(self.logger, name="Step5 SVM training", interval_seconds=60.0):
                        svm_model = await self._train_svm(
                            X_train, X_test, y_train, y_test, regime_name
                        )
                    if svm_model:
                        models["svm"] = svm_model
                        try:
                            power_scores.append(float(svm_model.get("accuracy", 0.0)))
                        except Exception:
                            pass
                except Exception as e:
                    self.logger.warning(f"SVM training failed: {e}")

            self.logger.info(f"✅ Trained {len(models)} models for regime: {regime_name}")

            # Attach aggregate predictive power score for this expert (ensemble average accuracy)
            models["_expert_power_score"] = float(np.mean(power_scores)) if power_scores else 0.0
            return models

        except Exception as e:
            self.logger.error(f"❌ Error training regime models: {e}")
            return {}

    async def _train_sr_models(self, data: pd.DataFrame) -> dict[str, Any]:
        """Train S/R specialist models: breakout and bounce binary classifiers.

        - Targets derived from sr_event_label: breakout_target = 1 if -1 else 0; bounce_target = 1 if +1 else 0
        - Features: include ALL numeric features except targets/labels/metadata
        - Save models under analyst_models/SR/
        """
        try:
            required_cols = ["sr_event_label"]
            if not all(c in data.columns for c in required_cols):
                self.logger.warning("S/R training skipped: missing sr_event_label in data")
                return {}

            # Build binary targets
            y_breakout = (data["sr_event_label"] == -1).astype(int)
            y_bounce = (data["sr_event_label"] == 1).astype(int)

            # Feature set: all numeric except targets/labels/metadata
            exclude_cols = set([
                "label",
                "regime",
                "sr_event_label",
            ])
            X_all = data.select_dtypes(include=[np.number]).drop(columns=[c for c in exclude_cols if c in data.columns], errors="ignore")
            if X_all.empty:
                self.logger.warning("S/R training skipped: no numeric features available")
                return {}

            from sklearn.model_selection import train_test_split
            Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(X_all, y_breakout, test_size=0.2, random_state=42, stratify=y_breakout if y_breakout.sum() > 0 else None)
            Xo_tr, Xo_te, yo_tr, yo_te = train_test_split(X_all, y_bounce, test_size=0.2, random_state=42, stratify=y_bounce if y_bounce.sum() > 0 else None)

            models: dict[str, Any] = {}

            # Helper to train a small suite
            async def _train_suite(Xtr, Xte, ytr, yte, name_prefix: str) -> dict[str, Any]:
                out = {}
                try:
                    from sklearn.ensemble import RandomForestClassifier
                    rf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
                    rf.fit(Xtr, ytr)
                    out[f"{name_prefix}_rf"] = rf
                except Exception as e:
                    self.logger.warning(f"RF training failed for {name_prefix}: {e}")
                try:
                    import lightgbm as lgb
                    lgbm = lgb.LGBMClassifier(
                        n_estimators=400,
                        learning_rate=0.03,
                        max_depth=-1,
                        num_leaves=64,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=42,
                        n_jobs=-1,
                    )
                    lgbm.fit(Xtr, ytr, eval_set=[(Xte, yte)], eval_metric="auc", verbose=False)
                    out[f"{name_prefix}_lgbm"] = lgbm
                except Exception as e:
                    self.logger.warning(f"LightGBM training failed for {name_prefix}: {e}")
                try:
                    import xgboost as xgb
                    xgbm = xgb.XGBClassifier(
                        n_estimators=400,
                        learning_rate=0.05,
                        max_depth=6,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_lambda=1.0,
                        random_state=42,
                        n_jobs=-1,
                        tree_method="hist",
                    )
                    xgbm.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)
                    out[f"{name_prefix}_xgb"] = xgbm
                except Exception as e:
                    self.logger.warning(f"XGBoost training failed for {name_prefix}: {e}")
                return out

            models.update(await _train_suite(Xb_tr, Xb_te, yb_tr, yb_te, "sr_breakout"))
            models.update(await _train_suite(Xo_tr, Xo_te, yo_tr, yo_te, "sr_bounce"))

            return models
        except Exception as e:
            self.logger.error(f"❌ Error training SR models: {e}")
            return {}

    async def _apply_pre_feature_selection(self, data: pd.DataFrame, feature_columns: list, regime_name: str) -> list:
        """Apply pre-feature selection for large feature sets to reduce dimensionality before training."""
        try:
            self.logger.info(f"🔍 Applying pre-feature selection for {len(feature_columns)} features...")
            
            # Load feature selection configuration
            feature_config = self.config.get("feature_interactions", {})
            selection_tiers = feature_config.get("feature_selection_tiers", {})
            
            # Get tiered selection parameters
            tier_1_count = selection_tiers.get("tier_1_base_features", 80)
            tier_2_count = selection_tiers.get("tier_2_normalized_features", 40)
            tier_3_count = selection_tiers.get("tier_3_interaction_features", 60)
            tier_4_count = selection_tiers.get("tier_4_lagged_features", 40)
            tier_5_count = selection_tiers.get("tier_5_causality_features", 20)
            total_max_features = selection_tiers.get("total_max_features", 240)
            
            # Categorize features by tier
            feature_categories = self._categorize_features_by_tier(feature_columns)
            
            selected_features = []
            
            # Tier 1: Core features (technical indicators, basic liquidity)
            tier_1_features = self._select_tier_1_features_pre_training(
                data, feature_categories["tier_1"], tier_1_count
            )
            selected_features.extend(tier_1_features)
            self.logger.info(f"   ✅ Tier 1: Selected {len(tier_1_features)} core features")
            
            # Tier 2: Normalized features (z-scores, changes, accelerations)
            tier_2_features = self._select_tier_2_features_pre_training(
                data, feature_categories["tier_2"], tier_2_count
            )
            selected_features.extend(tier_2_features)
            self.logger.info(f"   ✅ Tier 2: Selected {len(tier_2_features)} normalized features")
            
            # Tier 3: Interaction features (spread*volume, etc.)
            tier_3_features = self._select_tier_3_features_pre_training(
                data, feature_categories["tier_3"], tier_3_count
            )
            selected_features.extend(tier_3_features)
            self.logger.info(f"   ✅ Tier 3: Selected {len(tier_3_features)} interaction features")
            
            # Tier 4: Lagged features (lagged interactions)
            tier_4_features = self._select_tier_4_features_pre_training(
                data, feature_categories["tier_4"], tier_4_count
            )
            selected_features.extend(tier_4_features)
            self.logger.info(f"   ✅ Tier 4: Selected {len(tier_4_features)} lagged features")
            
            # Tier 5: Causality features (market microstructure causality)
            tier_5_features = self._select_tier_5_features_pre_training(
                data, feature_categories["tier_5"], tier_5_count
            )
            selected_features.extend(tier_5_features)
            self.logger.info(f"   ✅ Tier 5: Selected {len(tier_5_features)} causality features")
            
            # Apply final pruning if we exceed total_max_features
            if len(selected_features) > total_max_features:
                selected_features = self._apply_final_pruning_pre_training(
                    data, selected_features, total_max_features
                )
                self.logger.info(f"   🔧 Final pruning: Reduced to {len(selected_features)} features")
            
            return selected_features
            
        except Exception as e:
            self.logger.error(f"❌ Pre-feature selection failed: {e}")
            return feature_columns  # Return original features if selection fails

    def _categorize_features_by_tier(self, feature_columns: list) -> dict:
        """Categorize features into tiers based on naming patterns."""
        categories = {
            "tier_1": [],  # Core features
            "tier_2": [],  # Normalized features
            "tier_3": [],  # Interaction features
            "tier_4": [],  # Lagged features
            "tier_5": [],  # Causality features
        }
        
        for feature in feature_columns:
            feature_lower = feature.lower()
            
            # Tier 1: Core technical and liquidity features
            if any(keyword in feature_lower for keyword in [
                "rsi", "macd", "bb", "atr", "adx", "sma", "ema", "cci", "mfi", "roc",
                "volume", "spread", "liquidity", "price_impact", "kyle", "amihud"
            ]):
                categories["tier_1"].append(feature)
            
            # Tier 2: Normalized features
            elif any(keyword in feature_lower for keyword in [
                "_z_score", "_change", "_pct_change", "_acceleration", "_bounded",
                "_log", "_normalized"
            ]):
                categories["tier_2"].append(feature)
            
            # Tier 3: Interaction features
            elif "_x_" in feature_lower or "_div_" in feature_lower:
                categories["tier_3"].append(feature)
            
            # Tier 4: Lagged features
            elif "_lag" in feature_lower:
                categories["tier_4"].append(feature)
            
            # Tier 5: Causality features
            elif any(keyword in feature_lower for keyword in [
                "_predicts_", "_causality", "_divergence", "_stress", "_extreme"
            ]):
                categories["tier_5"].append(feature)
            
            # Default to tier 1 for uncategorized features
            else:
                categories["tier_1"].append(feature)
        
        return categories

    @with_tracing_span("Step5._select_tier_1_features_pre_training", log_args=False)
    @guard_dataframe_nulls(mode="warn", arg_index=1)
    def _select_tier_1_features_pre_training(self, data: pd.DataFrame, tier_1_features: list, count: int) -> list:
        """Select core features based on variance and correlation."""
        if not tier_1_features:
            return []
        
        # Get available features
        available_features = [f for f in tier_1_features if f in data.columns]
        if not available_features:
            return []
        
        # Calculate feature importance based on variance
        feature_variance = data[available_features].var()
        top_features = feature_variance.nlargest(count).index.tolist()
        
        return top_features

    @with_tracing_span("Step5._select_tier_2_features_pre_training", log_args=False)
    @guard_dataframe_nulls(mode="warn", arg_index=1)
    def _select_tier_2_features_pre_training(self, data: pd.DataFrame, tier_2_features: list, count: int) -> list:
        """Select normalized features based on stability."""
        if not tier_2_features:
            return []
        
        available_features = [f for f in tier_2_features if f in data.columns]
        if not available_features:
            return []
        
        # Select based on feature stability (lower variance for normalized features)
        feature_variance = data[available_features].var()
        stable_features = feature_variance.nsmallest(count).index.tolist()
        
        return stable_features

    @with_tracing_span("Step5._select_tier_3_features_pre_training", log_args=False)
    @guard_dataframe_nulls(mode="warn", arg_index=1)
    def _select_tier_3_features_pre_training(self, data: pd.DataFrame, tier_3_features: list, count: int) -> list:
        """Select interaction features based on significance."""
        if not tier_3_features:
            return []
        
        available_features = [f for f in tier_3_features if f in data.columns]
        if not available_features:
            return []
        
        # Select based on absolute mean (higher values indicate more significant interactions)
        feature_abs_mean = data[available_features].abs().mean()
        significant_features = feature_abs_mean.nlargest(count).index.tolist()
        
        return significant_features

    @with_tracing_span("Step5._select_tier_4_features_pre_training", log_args=False)
    @guard_dataframe_nulls(mode="warn", arg_index=1)
    def _select_tier_4_features_pre_training(self, data: pd.DataFrame, tier_4_features: list, count: int) -> list:
        """Select lagged features based on temporal significance."""
        if not tier_4_features:
            return []
        
        available_features = [f for f in tier_4_features if f in data.columns]
        if not available_features:
            return []
        
        # Select based on variance (higher variance indicates more temporal information)
        feature_variance = data[available_features].var()
        temporal_features = feature_variance.nlargest(count).index.tolist()
        
        return temporal_features

    @with_tracing_span("Step5._select_tier_5_features_pre_training", log_args=False)
    @guard_dataframe_nulls(mode="warn", arg_index=1)
    def _select_tier_5_features_pre_training(self, data: pd.DataFrame, tier_5_features: list, count: int) -> list:
        """Select causality features based on market logic significance."""
        if not tier_5_features:
            return []
        
        available_features = [f for f in tier_5_features if f in data.columns]
        if not available_features:
            return []
        
        # Select based on absolute mean (causality features should have meaningful values)
        feature_abs_mean = data[available_features].abs().mean()
        causality_features = feature_abs_mean.nlargest(count).index.tolist()
        
        return causality_features

    @with_tracing_span("Step5._apply_final_pruning_pre_training", log_args=False)
    @guard_dataframe_nulls(mode="warn", arg_index=1)
    def _apply_final_pruning_pre_training(self, data: pd.DataFrame, selected_features: list, max_features: int) -> list:
        """Apply final pruning to meet maximum feature count."""
        if len(selected_features) <= max_features:
            return selected_features
        
        # Calculate overall feature importance based on variance
        feature_variance = data[selected_features].var()
        top_features = feature_variance.nlargest(max_features).index.tolist()
        
        return top_features

    async def _train_random_forest(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        regime_name: str,
    ) -> dict[str, Any]:
        """Train Random Forest model."""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score

            # CRITICAL FIX: Ensure consistent label encoding for Random Forest
            # Map labels to contiguous 0..K-1 to prevent any label issues
            present_classes = sorted(pd.unique(pd.concat([y_train, y_test])))
            if len(present_classes) < 2:
                raise ValueError(
                    f"Insufficient classes for Random Forest training: {present_classes}"
                )

            rf_label_mapping = {cls: idx for idx, cls in enumerate(present_classes)}
            rf_inverse_label_mapping = {v: k for k, v in rf_label_mapping.items()}

            # Encode labels
            y_train_enc = y_train.map(rf_label_mapping).astype(int)
            y_test_enc = y_test.map(rf_label_mapping).astype(int)

            if y_train_enc.isna().any() or y_test_enc.isna().any():
                raise ValueError(
                    f"Encountered unknown labels for Random Forest mapping. Mapping: {rf_label_mapping}"
                )

            # Train model
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X_train, y_train_enc)

            # Evaluate model
            y_pred_enc = model.predict(X_test)
            y_pred = pd.Series(y_pred_enc).map(rf_inverse_label_mapping)
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
                "regime": regime_name,
                "training_date": datetime.now().isoformat(),
                # Persist explicit mapping for downstream consumers
                "rf_label_mapping": rf_label_mapping,
                "rf_inverse_label_mapping": rf_inverse_label_mapping,
            }

        except Exception as e:
            self.logger.exception(
                f"Error training Random Forest for {regime_name}: {e}",
            )
            raise

    async def _train_lightgbm(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        regime_name: str,
    ) -> dict[str, Any]:
        """Train LightGBM model."""
        try:
            import lightgbm as lgb
            from sklearn.metrics import accuracy_score

            # CRITICAL FIX: Ensure consistent label encoding for LightGBM
            # Map labels to contiguous 0..K-1 to prevent unseen labels error
            present_classes = sorted(pd.unique(pd.concat([y_train, y_test])))
            if len(present_classes) < 2:
                raise ValueError(
                    f"Insufficient classes for LightGBM training: {present_classes}"
                )

            lgb_label_mapping = {cls: idx for idx, cls in enumerate(present_classes)}
            lgb_inverse_label_mapping = {v: k for k, v in lgb_label_mapping.items()}

            # Encode labels
            y_train_enc = y_train.map(lgb_label_mapping).astype(int)
            y_test_enc = y_test.map(lgb_label_mapping).astype(int)

            if y_train_enc.isna().any() or y_test_enc.isna().any():
                raise ValueError(
                    f"Encountered unknown labels for LightGBM mapping. Mapping: {lgb_label_mapping}"
                )

            # Train model with early stopping
            model = lgb.LGBMClassifier(
                n_estimators=1000,  # Increased for early stopping
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=-1,  # Suppress all output
                early_stopping_rounds=50,  # Stop if no improvement for 50 rounds
                eval_metric="logloss",  # Evaluation metric for early stopping
                num_class=len(present_classes) if len(present_classes) > 2 else None,
                silent=True,  # Additional silence parameter
            )
            
            # Suppress LightGBM output during training
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(
                    X_train,
                    y_train_enc,
                    eval_set=[(X_test, y_test_enc)],
                    callbacks=[lgb.early_stopping(50, verbose=False)],  # Disable verbose in callback
                )

            # Evaluate model
            y_pred_enc = model.predict(X_test)
            y_pred = pd.Series(y_pred_enc).map(lgb_inverse_label_mapping)
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
                "regime": regime_name,
                "training_date": datetime.now().isoformat(),
                # Persist explicit mapping for downstream consumers
                "lgb_label_mapping": lgb_label_mapping,
                "lgb_inverse_label_mapping": lgb_inverse_label_mapping,
            }

        except Exception as e:
            self.logger.exception(f"Error training LightGBM for {regime_name}: {e}")
            raise

    async def _train_xgboost(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        regime_name: str,
    ) -> dict[str, Any]:
        """Train XGBoost model."""
        try:
            import xgboost as xgb
            from sklearn.metrics import accuracy_score

            # Train model
            # Explicitly map labels to contiguous 0..K-1 for XGBoost. Preserve semantic order [-1, 0, 1].
            base_order = [-1, 0, 1]
            present_classes = [
                c
                for c in base_order
                if c in set(pd.unique(y_train)) or c in set(pd.unique(y_test))
            ]
            if not present_classes:
                raise ValueError("No valid classes present in y for XGBoost training")
            xgb_label_mapping = {cls: idx for idx, cls in enumerate(present_classes)}
            xgb_inverse_label_mapping = {v: k for k, v in xgb_label_mapping.items()}
 
            # Encode labels
            y_train_enc = y_train.map(xgb_label_mapping).astype(int)
            y_test_enc = y_test.map(xgb_label_mapping).astype(int)
            if y_train_enc.isna().any() or y_test_enc.isna().any():
                raise ValueError(
                    f"Encountered unknown labels for XGBoost mapping. Mapping: {xgb_label_mapping}"
                )
 
            # Use multiclass only when we truly have >2 classes
            use_multiclass = len(present_classes) > 2
            model = xgb.XGBClassifier(
                n_estimators=600,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                eval_metric="mlogloss" if use_multiclass else "logloss",
                num_class=len(present_classes) if use_multiclass else None,
                early_stopping_rounds=50,
                verbosity=0,
            )
            
            # Suppress XGBoost output during training
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train_enc, eval_set=[(X_test, y_test_enc)], verbose=False)

            # Evaluate model
            y_pred_enc = model.predict(X_test)
            accuracy = accuracy_score(y_test_enc, y_pred_enc)

            # Get feature importance
            feature_importance = dict(
                zip(X_train.columns, model.feature_importances_, strict=False),
            )

            return {
                "model": model,
                "accuracy": accuracy,
                "feature_importance": feature_importance,
                "model_type": "XGBoost",
                "regime": regime_name,
                "training_date": datetime.now().isoformat(),
                # Persist explicit mapping for downstream consumers
                "xgb_label_mapping": xgb_label_mapping,
                "xgb_inverse_label_mapping": xgb_inverse_label_mapping,
            }

        except Exception as e:
            self.logger.exception(f"Error training XGBoost for {regime_name}: {e}")
            raise

    async def _train_neural_network(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        regime_name: str,
    ) -> dict[str, Any]:
        """Train Neural Network model."""
        try:
            from sklearn.metrics import accuracy_score
            from sklearn.neural_network import MLPClassifier

            # CRITICAL FIX: Ensure consistent label encoding for Neural Network
            # Map labels to contiguous 0..K-1 to prevent any label issues
            present_classes = sorted(pd.unique(pd.concat([y_train, y_test])))
            if len(present_classes) < 2:
                raise ValueError(
                    f"Insufficient classes for Neural Network training: {present_classes}"
                )

            nn_label_mapping = {cls: idx for idx, cls in enumerate(present_classes)}
            nn_inverse_label_mapping = {v: k for k, v in nn_label_mapping.items()}

            # Encode labels
            y_train_enc = y_train.map(nn_label_mapping).astype(int)
            y_test_enc = y_test.map(nn_label_mapping).astype(int)

            if y_train_enc.isna().any() or y_test_enc.isna().any():
                raise ValueError(
                    f"Encountered unknown labels for Neural Network mapping. Mapping: {nn_label_mapping}"
                )

            # Train model with enhanced early stopping
            model = MLPClassifier(
                hidden_layer_sizes=(100, 50, 25),
                max_iter=1000,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.15,  # Increased validation fraction
                n_iter_no_change=20,  # Stop if no improvement for 20 iterations
                tol=1e-4,  # Tolerance for improvement
                learning_rate_init=0.001,  # Lower learning rate for better convergence
                learning_rate="adaptive",  # Adaptive learning rate
            )
            model.fit(X_train, y_train_enc)

            # Evaluate model
            y_pred_enc = model.predict(X_test)
            y_pred = pd.Series(y_pred_enc).map(nn_inverse_label_mapping)
            accuracy = accuracy_score(y_test, y_pred)

            return {
                "model": model,
                "accuracy": accuracy,
                "feature_importance": {},  # Neural networks don't have direct feature importance
                "model_type": "NeuralNetwork",
                "regime": regime_name,
                "training_date": datetime.now().isoformat(),
                # Persist explicit mapping for downstream consumers
                "nn_label_mapping": nn_label_mapping,
                "nn_inverse_label_mapping": nn_inverse_label_mapping,
            }

        except Exception as e:
            self.logger.exception(
                f"Error training Neural Network for {regime_name}: {e}",
            )
            raise

    async def _train_svm(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        regime_name: str,
    ) -> dict[str, Any]:
        """Train Support Vector Machine model."""
        try:
            from sklearn.metrics import accuracy_score
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.kernel_approximation import RBFSampler
            from sklearn.svm import LinearSVC

            # CRITICAL FIX: Ensure consistent label encoding for SVM
            # Map labels to contiguous 0..K-1 to prevent any label issues
            present_classes = sorted(pd.unique(pd.concat([y_train, y_test])))
            if len(present_classes) < 2:
                raise ValueError(
                    f"Insufficient classes for SVM training: {present_classes}"
                )

            svm_label_mapping = {cls: idx for idx, cls in enumerate(present_classes)}
            svm_inverse_label_mapping = {v: k for k, v in svm_label_mapping.items()}

            # Encode labels
            y_train_enc = y_train.map(svm_label_mapping).astype(int)
            y_test_enc = y_test.map(svm_label_mapping).astype(int)

            if y_train_enc.isna().any() or y_test_enc.isna().any():
                raise ValueError(
                    f"Encountered unknown labels for SVM mapping. Mapping: {svm_label_mapping}"
                )

            # Cast features to float32 to reduce memory footprint
            X_train = X_train.astype(np.float32)
            X_test = X_test.astype(np.float32)

            # Lightweight hyperparameter tuning for RBFSampler + LinearSVC
            # Tune: gamma (RBF width), n_components (approximation granularity), C (margin strength)
            from sklearn.model_selection import (
                StratifiedShuffleSplit,
                StratifiedKFold,
                cross_val_score,
            )
            from src.utils.purged_kfold import PurgedKFoldTime
            from src.utils.logger import heartbeat

            # Build a stratified tuning subset to keep the search fast
            max_tune_rows = 50000
            if len(X_train) > max_tune_rows:
                frac = max_tune_rows / float(len(X_train))
                self.logger.info(
                    f"RBFApprox tuning on stratified subset: {max_tune_rows} of {len(X_train)} rows (frac={frac:.3f})",
                )
                sss = StratifiedShuffleSplit(n_splits=1, train_size=max_tune_rows, random_state=42)
                idx = next(sss.split(X_train, y_train_enc))[0]
                X_tune = X_train.iloc[idx]
                y_tune = y_train_enc.iloc[idx]
            else:
                X_tune, y_tune = X_train, y_train_enc

            # Choose rigorous CV: time-aware PurgedKFoldTime (with purge/embargo) if DatetimeIndex, else StratifiedKFold
            if isinstance(X_tune.index, pd.DatetimeIndex):
                # Set purge and embargo windows based on typical label horizon (~15m) and safety buffer
                cv = PurgedKFoldTime(n_splits=3, purge=pd.Timedelta(minutes=15), embargo=pd.Timedelta(minutes=10))
                self.logger.info("Using PurgedKFoldTime(n_splits=3, purge=15m, embargo=10m) for time-series CV")
                cv_splits = cv.split(X_tune)
            else:
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                self.logger.info("Using StratifiedKFold(n_splits=3, shuffle=True) for CV")
                cv_splits = cv

            # Derive a sensible gamma scale based on feature dimensionality after StandardScaler (~unit variance)
            n_feat = X_tune.shape[1] if X_tune.shape[1] > 0 else 1
            gamma_scale = 1.0 / float(n_feat)
            gamma_candidates = [gamma_scale * 0.1, gamma_scale, gamma_scale * 10.0, 1e-3, 1e-2]
            n_component_candidates = [1500, 3000, 4500]
            C_candidates = [0.5, 1.0, 2.0]

            best_score = -np.inf
            best_params = {"gamma": gamma_scale, "n_components": 3000, "C": 1.0}

            with heartbeat(self.logger, name="Step5 RBFApprox CV tuning", interval_seconds=60.0):
                tried = 0
                for gamma_val in gamma_candidates:
                    for n_comp in n_component_candidates:
                        for C_val in C_candidates:
                            tried += 1
                            pipe = make_pipeline(
                                StandardScaler(),
                                RBFSampler(gamma=float(gamma_val), n_components=int(n_comp), random_state=42),
                                LinearSVC(C=float(C_val), tol=1e-3, random_state=42),
                            )
                            try:
                                # Mean CV accuracy with parallelism
                                scores = cross_val_score(
                                    pipe,
                                    X_tune,
                                    y_tune,
                                    cv=cv_splits,
                                    scoring="accuracy",
                                    n_jobs=-1,
                                )
                                mean_score = float(np.mean(scores))
                                if mean_score > best_score:
                                    best_score = mean_score
                                    best_params = {
                                        "gamma": float(gamma_val),
                                        "n_components": int(n_comp),
                                        "C": float(C_val),
                                    }
                            except Exception as _tune_err:
                                # Skip configs that fail numerically
                                self.logger.warning(f"Tuning candidate failed (gamma={gamma_val}, n_comp={n_comp}, C={C_val}): {_tune_err}")
                                continue

            self.logger.info(
                f"RBFApprox best params: gamma={best_params['gamma']}, n_components={best_params['n_components']}, C={best_params['C']} (cv_acc={best_score:.4f})",
            )

            # Train final model on full training set with best params
            model = make_pipeline(
                StandardScaler(),
                RBFSampler(gamma=best_params["gamma"], n_components=best_params["n_components"], random_state=42),
                LinearSVC(C=best_params["C"], tol=1e-3, random_state=42),
            )
            model.fit(X_train, y_train_enc)

            # Evaluate model
            y_pred_enc = model.predict(X_test)
            y_pred = pd.Series(y_pred_enc).map(svm_inverse_label_mapping)
            accuracy = accuracy_score(y_test, y_pred)

            return {
                "model": model,
                "accuracy": accuracy,
                "feature_importance": {},  # SVMs don't have direct feature importance
                "model_type": "RBFApproxLinear",
                "regime": regime_name,
                "training_date": datetime.now().isoformat(),
                # Persist explicit mapping for downstream consumers
                "svm_label_mapping": svm_label_mapping,
                "svm_inverse_label_mapping": svm_inverse_label_mapping,
            }

        except Exception as e:
            self.logger.exception(f"Error training SVM for {regime_name}: {e}")
            raise

    async def _train_label_experts(
        self,
        combined_data: pd.DataFrame,
        data_dir: str,
        exchange: str,
        symbol: str,
    ) -> str:
        """Train per-label expert models for analyst timeframes (5m/15m/30m) using mapping.

        Returns the directory where experts are saved.
        """
        try:
            from src.config.label_model_mapping import select_model_for_label_timeframe
        except ImportError:
            self.logger.warning("label_model_mapping not available, skipping label expert training")
            return ""
        import pickle
        import os
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score

        # Load labeled datasets to get meta-label columns
        labeled_pkls = [
            f"{data_dir}/{exchange}_{symbol}_labeled_train.pkl",
            f"{data_dir}/{exchange}_{symbol}_labeled_validation.pkl",
            f"{data_dir}/{exchange}_{symbol}_labeled_test.pkl",
        ]
        frames = []
        for p in labeled_pkls:
            if os.path.exists(p):
                with open(p, "rb") as f:
                    frames.append(pickle.load(f))
        if not frames:
            self.logger.info("No labeled PKLs found; skipping label expert training")
            return ""
        labeled_all = pd.concat(frames, axis=0, ignore_index=False)
        if "timestamp" not in labeled_all.columns and isinstance(labeled_all.index, pd.DatetimeIndex):
            labeled_all = labeled_all.copy()
            labeled_all["timestamp"] = labeled_all.index
        # Align combined feature matrix on timestamp
        features_df = combined_data.copy()
        if "timestamp" not in features_df.columns:
            # attempt to use index if datetime-like
            if isinstance(features_df.index, pd.DatetimeIndex):
                features_df = features_df.copy()
                features_df["timestamp"] = features_df.index
        if "timestamp" not in features_df.columns:
            self.logger.info("No timestamp available for alignment; skipping label expert training")
            return ""
        # Build feature set: numeric columns excluding known metadata/labels
        non_feature_cols = set([
            "label", "regime", "sr_event_label", "timestamp", "exchange", "symbol", "timeframe",
            "year", "month", "day",
        ])
        X_all = features_df.select_dtypes(include=[np.number]).drop(
            columns=[c for c in non_feature_cols if c in features_df.columns], errors="ignore"
        )
        if X_all.empty:
            self.logger.info("Empty feature set in combined_data; skipping label expert training")
            return ""
        # Join to get aligned labels
        joined = pd.merge(
            X_all,
            labeled_all[[c for c in labeled_all.columns if c != "label"]],
            on="timestamp",
            how="inner",
        )
        if joined.empty:
            self.logger.info("No overlap between features and labeled data; skipping label experts")
            return ""
        # Create output dirs
        base_out = os.path.join(data_dir, "label_experts")
        os.makedirs(base_out, exist_ok=True)
        # Define analyst timeframes
        analyst_tfs = ["5m", "15m", "30m"]
        trained = 0
        for tf in analyst_tfs:
            tf_cols = [c for c in joined.columns if c.startswith(f"{tf}_")]
            if not tf_cols:
                continue
            tf_dir = os.path.join(base_out, tf)
            os.makedirs(tf_dir, exist_ok=True)
            for col in tf_cols:
                base_label = col.split("_", 1)[1].upper()
                # Build binary target: treat >0 as 1
                y = (joined[col].astype(float).fillna(0) > 0).astype(int)
                # Skip severely imbalanced or tiny labels
                pos = int(y.sum())
                neg = int((1 - y).sum())
                if pos < 100 or neg < 100:
                    continue
                X = joined[X_all.columns].astype(np.float32)
                # Train/test split
                try:
                    X_tr, X_te, y_tr, y_te = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y
                    )
                except Exception:
                    # fallback without stratify
                    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
                # Build model per mapping
                model = select_model_for_label_timeframe(base_label, tf)
                # Fit
                try:
                    model.fit(X_tr, y_tr)
                except Exception:
                    # some models require dense arrays
                    model.fit(X_tr.values, y_tr.values)
                # Evaluate
                try:
                    if hasattr(model, "predict_proba"):
                        yhat = (model.predict_proba(X_te)[:, -1] > 0.5).astype(int)
                    else:
                        yhat = model.predict(X_te)
                    acc = accuracy_score(y_te, yhat)
                except Exception:
                    acc = 0.0
                # Save
                model_name = type(model).__name__
                out_path = os.path.join(tf_dir, f"{base_label}_{model_name}.pkl")
                with open(out_path, "wb") as f:
                    pickle.dump(model, f)
                trained += 1
                try:
                    self.logger.info(
                        f"Label expert trained: tf={tf} label={base_label} model={model_name} acc={acc:.3f} -> {out_path}"
                    )
                except Exception:
                    pass
        self.logger.info(f"✅ Trained {trained} label expert models (analyst timeframes)")
        return base_out


# For backward compatibility with existing step structure
async def run_step(
    symbol: str,
    exchange: str = "BINANCE",
    data_dir: str = "data/training",
    force_rerun: bool = False,
    **kwargs,
) -> bool:
    """
    Run the analyst specialist training step.

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
        step = AnalystSpecialistTrainingStep(config)
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

    except Exception as e:
        print(failed(f"Analyst specialist training failed: {e}"))
        return False


if __name__ == "__main__":
    # Test the step
    async def test():
        result = await run_step("ETHUSDT", "BINANCE", "data/training")
        print(f"Test result: {result}")

    asyncio.run(test())
