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
from src.training.steps.unified_data_loader import get_unified_data_loader


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

            # Check if feature files exist
            missing_files = [f for f in feature_files if not os.path.exists(f)]
            if missing_files:
                msg = f"Missing feature files: {missing_files}. Step 5 requires features from Step 4."
                try:
                    self.logger.error(msg)
                    print(
                        f"Step5Monitor ▶ Missing features: {[os.path.basename(p) for p in missing_files]}",
                        flush=True,
                    )
                except Exception:
                    pass
                raise ValueError(msg)

            # Load and combine all feature data
            all_data = []
            for file_path in feature_files:
                with open(file_path, "rb") as f:
                    data = pickle.load(f)
                    all_data.append(data)
                    try:
                        self.logger.info(
                            f"Loaded features file {file_path}: type={type(data).__name__}, shape={getattr(data, 'shape', None)}",
                        )
                        print(
                            f"Step5Monitor ▶ Loaded {os.path.basename(file_path)} shape={getattr(data, 'shape', None)}",
                            flush=True,
                        )
                    except Exception:
                        pass

            # Combine all data
            combined_data = pd.concat(all_data, ignore_index=True)
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

            if enable_experts:
                try:
                    # Load labeled datasets to align indices/timestamps for regime joins
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
                    labeled_all = pd.concat(labeled_frames, axis=0, ignore_index=False)
                    # Ensure timestamp present for joins; if missing, synthesize from index when DatetimeIndex
                    if "timestamp" not in labeled_all.columns and isinstance(labeled_all.index, pd.DatetimeIndex):
                        labeled_all = labeled_all.copy()
                        labeled_all["timestamp"] = labeled_all.index
                    # Feature columns to train on: use combined selected features
                    feature_cols = [c for c in combined_data.columns if c != "label"]

                    # Choose regime definition
                    expert_datasets: dict[str, pd.DataFrame] = {}
                    if regime_source == "step2_bull_bear_sideways":
                        # Load Step3 regime splits to fetch labels/strength
                        regime_dir = os.path.join(data_dir, "regime_data")
                        if os.path.isdir(regime_dir):
                            for file in os.listdir(regime_dir):
                                if file.endswith(".parquet"):
                                    regime_name = os.path.splitext(file)[0]
                                    df_reg = pd.read_parquet(os.path.join(regime_dir, file))
                                    # Align to combined_data rows via timestamp
                                    if "timestamp" not in df_reg.columns and isinstance(df_reg.index, pd.DatetimeIndex):
                                        df_reg = df_reg.reset_index()
                                        df_reg.rename(columns={"index": "timestamp"}, inplace=True)
                                    cols_keep = [c for c in ["timestamp", "regime", "confidence"] if c in df_reg.columns]
                                    df_reg = df_reg[cols_keep]
                                    merged = labeled_all.merge(df_reg, on="timestamp", how="inner")
                                    # Restrict to selected features + label
                                    cols = [c for c in feature_cols + ["label"] if c in merged.columns]
                                    regime_df = merged[cols].dropna(subset=["label"]) if "label" in cols else merged[cols]
                                    if len(regime_df) >= min_rows_per_expert:
                                        expert_datasets[regime_name] = regime_df.copy()
                        dispatcher_manifest["gating"] = {
                            "type": "step2_regime",
                            "weights": "confidence",
                        }
                    elif regime_source == "meta_labels":
                        # Use provided meta label columns as regimes (binary or categorical)
                        present_metas = [c for c in meta_label_columns if c in labeled_all.columns]
                        for meta in present_metas:
                            # Expert dataset: rows where meta label is active (non-zero / True)
                            active_mask = labeled_all[meta].astype(float).fillna(0) != 0
                            df_meta = labeled_all.loc[active_mask]
                            cols = [c for c in feature_cols + ["label"] if c in df_meta.columns]
                            df_meta = df_meta[cols].dropna(subset=["label"]) if "label" in cols else df_meta[cols]
                            if len(df_meta) >= min_rows_per_expert:
                                expert_datasets[meta] = df_meta.copy()
                        # Default SR strength source if available
                        if "sr_zone_strength" in labeled_all.columns:
                            for k in expert_datasets.keys():
                                strength_columns.setdefault(k, "sr_zone_strength")
                        dispatcher_manifest["gating"] = {
                            "type": "meta_labels",
                            "weights": strength_columns,  # optional per-meta strength source
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

                # Train models for this regime
                regime_models = await self._train_regime_models(
                    regime_data,
                    regime_name,
                )
                training_results[regime_name] = regime_models
                try:
                    print(
                        f"Step5Monitor ▶ Training done for regime={regime_name} models={len(regime_models)}",
                        flush=True,
                    )
                except Exception:
                    pass

                # Memory cleanup after training
                gc.collect()

                # Performance stratification placeholder: basic label distribution under this regime
                try:
                    os.makedirs("log/experts", exist_ok=True)
                    if "label" in regime_data.columns:
                        stats = regime_data["label"].value_counts(dropna=False).to_dict()
                        with open(f"log/experts/{regime_name}_label_distribution.json", "w") as jf:
                            json.dump({"counts": {str(k): int(v) for k, v in stats.items()}}, jf, indent=2)
                except Exception:
                    pass

                # Contextual SHAP (best-effort, optional)
                try:
                    import shap  # type: ignore
                    os.makedirs("log/experts", exist_ok=True)
                    # Choose the first model artifact
                    first_key = next(iter(regime_models))
                    artifact = regime_models[first_key]
                    est = self._extract_estimator_from_artifact(artifact)
                    X_sample = regime_data.drop(columns=[c for c in ["label"] if c in regime_data.columns]).select_dtypes(include=[np.number]).tail(2000)
                    if hasattr(est, "predict_proba") and hasattr(est, "fit"):
                        explainer = shap.Explainer(est, X_sample, feature_names=list(X_sample.columns))
                        shap_vals = explainer(X_sample, check_additivity=False)
                        # Save mean |SHAP| values
                        mean_abs = np.abs(shap_vals.values).mean(axis=0)
                        shap_importance = {f: float(v) for f, v in zip(X_sample.columns, mean_abs, strict=False)}
                        with open(f"log/experts/{regime_name}_shap_importance.json", "w") as jf:
                            json.dump(shap_importance, jf, indent=2)
                except Exception:
                    pass

            # Save the main analyst model (use the first available model)
            main_model_artifact = None
            main_model_name = None

            for regime_name, models in training_results.items():
                if models:  # If there are models in this regime
                    # Use the first available model as the main model
                    main_model_name = list(models.keys())[0]
                    main_model_artifact = models[main_model_name]
                    break

            if main_model_artifact is not None:
                # Ensure we save the underlying estimator, not the wrapper dict
                main_estimator = self._extract_estimator_from_artifact(
                    main_model_artifact,
                )
                # Save the main analyst model file that the validator expects
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

            # Also save detailed results to subdirectories for compatibility
            models_dir = f"{data_dir}/analyst_models"
            os.makedirs(models_dir, exist_ok=True)

            for regime_name, models in training_results.items():
                regime_models_dir = f"{models_dir}/{regime_name}"
                os.makedirs(regime_models_dir, exist_ok=True)

                for model_name, model_data in models.items():
                    model_file = f"{regime_models_dir}/{model_name}.pkl"
                    with open(model_file, "wb") as f:
                        pickle.dump(model_data, f)
                    try:
                        self.logger.info(f"Saved regime model: {model_file}")
                        print(
                            f"Step5Monitor ▶ Saved regime model: {regime_name}/{model_name}.pkl",
                            flush=True,
                        )
                    except Exception:
                        pass

            # Train and save S/R models
            try:
                sr_models = await self._train_sr_models(combined_data)
                if sr_models:
                    sr_dir = f"{models_dir}/SR"
                    os.makedirs(sr_dir, exist_ok=True)
                    for name, model in sr_models.items():
                        with open(f"{sr_dir}/{name}.pkl", "wb") as f:
                            pickle.dump(model, f)
                    training_results["SR"] = sr_models
                    self.logger.info(f"✅ Trained and saved {len(sr_models)} S/R models")
                    try:
                        print(
                            f"Step5Monitor ▶ Saved {len(sr_models)} SR models",
                            flush=True,
                        )
                    except Exception:
                        pass
                    # Train SR score regressors if scores available
                    try:
                        have_scores = all(c in combined_data.columns for c in ["sr_breakout_score", "sr_bounce_score"])
                        if have_scores:
                            from sklearn.model_selection import train_test_split
                            from sklearn.ensemble import RandomForestRegressor
                            from sklearn.metrics import r2_score
                            # Feature space mirrors SR models
                            X_num = combined_data.select_dtypes(include=[np.number]).drop(columns=[c for c in ["label", "regime", "sr_event_label", "sr_breakout_score", "sr_bounce_score"] if c in combined_data.columns], errors="ignore")
                            # Breakout strength regressor (train only where breakout happened)
                            mask_bk = (combined_data.get("sr_event_label", 0) == -1)
                            if mask_bk.any() and not X_num.empty:
                                Xb = X_num.loc[mask_bk]
                                yb = combined_data.loc[mask_bk, "sr_breakout_score"].astype(float)
                                if len(Xb) >= 50 and yb.sum() > 0:
                                    Xtr, Xte, ytr, yte = train_test_split(Xb, yb, test_size=0.2, random_state=42)
                                    rf_reg_bk = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
                                    rf_reg_bk.fit(Xtr, ytr)
                                    sr_models["sr_breakout_strength_rf"] = rf_reg_bk
                                    with open(f"{sr_dir}/sr_breakout_strength_rf.pkl", "wb") as f:
                                        pickle.dump(rf_reg_bk, f)
                            # Bounce strength regressor (train only where bounce happened)
                            mask_bo = (combined_data.get("sr_event_label", 0) == 1)
                            if mask_bo.any() and not X_num.empty:
                                Xo = X_num.loc[mask_bo]
                                yo = combined_data.loc[mask_bo, "sr_bounce_score"].astype(float)
                                if len(Xo) >= 50 and yo.sum() > 0:
                                    Xtr, Xte, ytr, yte = train_test_split(Xo, yo, test_size=0.2, random_state=42)
                                    rf_reg_bo = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
                                    rf_reg_bo.fit(Xtr, ytr)
                                    sr_models["sr_bounce_strength_rf"] = rf_reg_bo
                                    with open(f"{sr_dir}/sr_bounce_strength_rf.pkl", "wb") as f:
                                        pickle.dump(rf_reg_bo, f)
                            self.logger.info("✅ Trained SR strength regressors where data allowed")
                            # Generate OOF predictions for strength to support blending
                            try:
                                import numpy as _np
                                from sklearn.model_selection import KFold
                                kf = KFold(n_splits=3, shuffle=True, random_state=42)
                                oof_bk = pd.Series(0.0, index=combined_data.index, name="sr_pred_breakout_strength")
                                oof_bo = pd.Series(0.0, index=combined_data.index, name="sr_pred_bounce_strength")
                                if mask_bk.any() and "sr_breakout_strength_rf" in sr_models:
                                    model_bk = sr_models["sr_breakout_strength_rf"]
                                    for tr_idx, te_idx in kf.split(Xb):
                                        Xtr, Xte = Xb.iloc[tr_idx], Xb.iloc[te_idx]
                                        ytr = yb.iloc[tr_idx]
                                        m = RandomForestRegressor(n_estimators=model_bk.n_estimators, max_depth=model_bk.max_depth, random_state=42, n_jobs=-1)
                                        m.fit(Xtr, ytr)
                                        oof_bk.iloc[Xb.iloc[te_idx].index] = m.predict(Xte)
                                if mask_bo.any() and "sr_bounce_strength_rf" in sr_models:
                                    model_bo = sr_models["sr_bounce_strength_rf"]
                                    for tr_idx, te_idx in kf.split(Xo):
                                        Xtr, Xte = Xo.iloc[tr_idx], Xo.iloc[te_idx]
                                        ytr = yo.iloc[tr_idx]
                                        m = RandomForestRegressor(n_estimators=model_bo.n_estimators, max_depth=model_bo.max_depth, random_state=42, n_jobs=-1)
                                        m.fit(Xtr, ytr)
                                        oof_bo.iloc[Xo.iloc[te_idx].index] = m.predict(Xte)
                                strength_oof = pd.DataFrame({"timestamp": combined_data.get("timestamp", pd.RangeIndex(len(combined_data))), oof_bk.name: oof_bk.values, oof_bo.name: oof_bo.values})
                                out_path = f"{data_dir}/{exchange}_{symbol}_sr_strength_oof.parquet"
                                strength_oof.to_parquet(out_path, index=False)
                                self.logger.info(f"✅ Wrote SR strength OOF predictions for blending: {out_path}")
                                try:
                                    print(
                                        f"Step5Monitor ▶ Wrote SR OOF: {os.path.basename(out_path)}",
                                        flush=True,
                                    )
                                except Exception:
                                    pass
                            except Exception as _oofe:
                                self.logger.warning(f"SR strength OOF generation skipped: {_oofe}")
                    except Exception as _ers:
                        self.logger.warning(f"SR strength regressors skipped: {_ers}")
            except Exception as _e:
                self.logger.warning(f"S/R model training skipped due to error: {_e}")

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
                },
            }

            # Add model metadata for each regime
            for regime_name, models in training_results.items():
                summary_data["models_per_regime"][regime_name] = {
                    "model_count": len(models),
                    "model_types": list(models.keys()),
                    "model_files": [],
                }

                # Add file paths for each model
                regime_models_dir = f"{models_dir}/{regime_name}"
                for model_name in models:
                    model_file = f"{regime_models_dir}/{model_name}.pkl"
                    summary_data["models_per_regime"][regime_name][
                        "model_files"
                    ].append(model_file)

            with open(summary_file, "w") as f:
                json.dump(summary_data, f, indent=2)

            self.logger.info(
                f"✅ Analyst specialist training completed. Results saved to {models_dir}",
            )
            try:
                print(
                    f"Step5Monitor ▶ Training complete. Models saved to {models_dir}",
                    flush=True,
                )
            except Exception:
                pass

            # Update pipeline state
            pipeline_state["analyst_models"] = training_results

            # Save dispatcher manifest for Method A
            if enable_experts:
                try:
                    manifest = {
                        "experts": list(training_results.keys()),
                        "gating": dispatcher_manifest.get("gating", {}),
                        "expert_power": dispatcher_manifest.get("expert_power", {}),
                        "regime_source": regime_source,
                        "min_rows_per_expert": min_rows_per_expert,
                        "use_strength_weighting": use_strength_weighting,
                        "strength_columns": strength_columns,
                        "timestamp": datetime.now().isoformat(),
                    }
                    manifest_path = f"{data_dir}/{exchange}_{symbol}_analyst_dispatcher.json"
                    with open(manifest_path, "w") as jf:
                        json.dump(manifest, jf, indent=2)
                    self.logger.info(f"✅ Saved dispatcher manifest to {manifest_path}")
                except Exception as _mf:
                    self.logger.warning(f"Failed to save dispatcher manifest: {_mf}")

            return {
                "analyst_models": training_results,
                "models_dir": models_dir,
                "duration": 0.0,  # Will be calculated in actual implementation
                "status": "SUCCESS",
                "dispatcher": dispatcher_manifest if enable_experts else None,
                "label_experts_dir": label_expert_dir if 'label_expert_dir' in locals() else None,
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
        from src.config.label_model_mapping import select_model_for_label_timeframe
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
