"""
ML Volume Force Step

This step constructs Volume Force/Impulse features and trains an XGBoost classifier
to predict market direction (Down/Neutral/Up) based on order flow pressure,
volume deltas, and shocks.

Primary goals:
- Generate volume force/impulse features (e.g., Volume Delta, Force Index, Kyle's Lambda).
- Train an XGBoost model using StandardizedXGBTrainer.
- Predict 2-3h forward returns (8-12 bars at 15m).
- Output a scalar directional prediction (0=Strong Down, 0.5=Neutral, 1=Strong Up).
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score, brier_score_loss

from src.training.steps.base_step import BaseStep
from src.utils.tprint import (
    tprint,
    tprint_info,
    tprint_warning,
    tprint_error,
    tprint_success,
)
from src.utils.versioned_artifacts.temporal_splits import (
    create_temporal_split_config_for_pipeline,
    TemporalSplitConfig,
)
from src.utils.ml_common.standardized_xgb_trainer import (
    StandardizedXGBTrainer,
    XGBTrainingConfig,
)
from src.utils.ml_common.retraining_scheduler import create_sample_weights
from src.feature_generation.categories.volume_force_features import (
    generate_volume_force_features,
)
from src.feature_generation.categories.liquidity_regime_features import (
    generate_liquidity_regime_features,
)

logger = logging.getLogger(__name__)


class MLVolumeForceStep(BaseStep):
    """Pipeline step for Volume Force/Impulse analysis and prediction."""

    def __init__(self, step_name: str = "ml_volume_force_step"):
        super().__init__(step_name, use_versioned_artifacts=True)
        self.logger = logger.getChild("MLVolumeForceStep") if hasattr(logger, "getChild") else logger
        self._cached_market_data = None
        self._cached_market_source = None
        self._cached_market_cache_key = None
        # Cache for feature generation in batch mode
        self._feature_cache = {}
        tprint(f"✅ Initialized {step_name} step", "SUCCESS")

    async def run_config_batch(
        self, configs: List[Dict[str, Any]], symbol: str, exchange: str
    ) -> List[Dict[str, Any]]:
        """Run a batch of configurations and collect results."""
        results = []
        total_configs = len(configs)

        for i, config in enumerate(configs):
            # Ensure symbol/exchange are set
            config["symbol"] = symbol
            config["exchange"] = exchange
            config["execution_mode"] = config.get("execution_mode", "light")

            # Enable batch mode flags if useful for speed
            config["is_batch_run"] = True

            tprint_info(
                f"🚀 Running config {i+1}/{total_configs}: {self.get_config_signature(config)}"
            )

            try:
                start_time = time.time()
                result = await self.execute(config)
                execution_time = time.time() - start_time

                metrics = result.get("metrics", {})

                # Extract key metrics
                run_metrics = {
                    "config_id": i + 1,
                    "config_signature": self.get_config_signature(config),
                    "execution_time": execution_time,
                    "success": result.get("success", False),
                    "error": result.get("error", ""),

                    # Performance Metrics
                    "oof_log_loss": metrics.get("oof_log_loss", float("inf")),
                    "oof_accuracy": metrics.get("accuracy", 0.0),
                    "scalar_pred_mean": metrics.get("scalar_pred_mean", 0.0),
                    "scalar_pred_std": metrics.get("scalar_pred_std", 0.0),

                    # Data Stats
                    "n_samples": metrics.get("n_samples", 0),
                    "class_balance": metrics.get("class_distribution", {}),
                    "n_oof_samples": metrics.get("n_oof_samples", 0),
                    "oof_start": metrics.get("oof_start"),
                    "oof_end": metrics.get("oof_end"),
                }

                # Add config params
                run_metrics.update({
                    f"config_{k}": v for k, v in config.items()
                    if k.startswith("volume_force_")
                })

                results.append(run_metrics)

                if result.get("success", False):
                    tprint_success(
                        f"✅ Config {i+1} done. Loss: {run_metrics['oof_log_loss']:.4f}, "
                        f"Acc: {run_metrics['oof_accuracy']:.4f}"
                    )
                else:
                    tprint_warning(f"⚠️ Config {i+1} failed: {run_metrics['error']}")

            except Exception as e:
                tprint_error(f"❌ Config {i+1} crashed: {e}")
                results.append({
                    "config_id": i + 1,
                    "config_signature": self.get_config_signature(config),
                    "success": False,
                    "error": str(e),
                    "oof_log_loss": float("inf")
                })

        return results

    def analyze_and_rank_results(
        self, results: List[Dict[str, Any]]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Analyze results and rank configurations."""
        if not results:
            return pd.DataFrame(), {}

        df = pd.DataFrame(results)

        # Filter successful runs
        successful = df[df["success"] == True].copy()

        if successful.empty:
            return df, {"best_config": None, "analysis": "no_successful_runs"}

        # ------------------------------------------------------------------
        # Quality filters for noisy, low-SNR classifier
        # ------------------------------------------------------------------
        N_MIN = 300
        N_OOF_MIN = 300
        ENTROPY_MIN = 0.8

        # Compute class entropy from class_balance dicts
        def _class_entropy(cb: Any) -> float:
            try:
                if not cb:
                    return float("nan")
                import numpy as _np
                counts = _np.array(list(cb.values()), dtype=float)
                total = counts.sum()
                if total <= 0:
                    return float("nan")
                p = counts / total
                p = p[p > 0]
                return float(-(p * _np.log(p)).sum())
            except Exception:
                return float("nan")

        if "class_balance" in successful.columns:
            successful["class_entropy"] = successful["class_balance"].apply(_class_entropy)

        # Ensure auxiliary columns exist
        if "n_oof_samples" not in successful.columns:
            successful["n_oof_samples"] = 0

        # Apply filters
        successful = successful[successful["n_samples"] >= N_MIN]
        if "class_entropy" in successful.columns:
            successful = successful[successful["class_entropy"] >= ENTROPY_MIN]
        successful = successful[successful["n_oof_samples"] >= N_OOF_MIN]

        if successful.empty:
            return df, {"best_config": None, "analysis": "no_successful_runs_after_filters"}

        # Compute OOF duration (in days) where available
        import pandas as _pd

        def _oof_duration_days(row: Any) -> float:
            try:
                start = row.get("oof_start")
                end = row.get("oof_end")
                if start is None or end is None:
                    return 0.0
                start_ts = _pd.to_datetime(start)
                end_ts = _pd.to_datetime(end)
                if _pd.isna(start_ts) or _pd.isna(end_ts):
                    return 0.0
                delta = end_ts - start_ts
                return float(delta.days) + float(delta.seconds) / 86400.0
            except Exception:
                return 0.0

        successful["oof_duration_days"] = successful.apply(_oof_duration_days, axis=1)

        # Sort by Log Loss (lower is better), then by longer OOF duration
        successful = successful.sort_values(
            ["oof_log_loss", "oof_duration_days"], ascending=[True, False]
        )

        # Get best config
        best_run = successful.iloc[0].to_dict()

        # Construct best config dict (stripping 'config_' prefix)
        best_config = {}
        for k, v in best_run.items():
            if k.startswith("config_"):
                best_config[k.replace("config_", "")] = v

        analysis = {
            "best_config": best_config,
            "best_log_loss": best_run["oof_log_loss"],
            "best_accuracy": best_run["oof_accuracy"],
            "total_runs": len(results),
            "successful_runs": len(successful),
        }

        return df, analysis

    def get_config_signature(self, config: Dict[str, Any]) -> str:
        """Generate a compact signature for configuration identification."""
        keys = [
            "volume_force_target_threshold_atr",
            "volume_force_lookahead",
            "volume_force_normalization_window"
        ]
        parts = []
        for k in keys:
            if k in config:
                val = config[k]
                parts.append(f"{k.replace('volume_force_', '')}={val}")
        return "|".join(parts)

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the volume force step."""
        start_time = time.time()
        metrics: Dict[str, Any] = {}
        artifacts: List[str] = []

        try:
            symbol = str(config.get("symbol", "ETHUSDT"))
            exchange = str(config.get("exchange", "binance"))
            timeframe = str(config.get("timeframe", "15m"))
            direction = str(config.get("direction", "long"))

            if not symbol or not exchange:
                raise ValueError("Config must include 'symbol' and 'exchange'")

            self.set_context(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                direction=direction,
                model="volume_force",
            )

            tprint_info(
                f"🚀 Starting {self.step_name} for {symbol} on {exchange} "
                f"(timeframe={timeframe})"
            )

            # 1. Load Market Data
            market_data, market_source = self._load_market_data_with_cache(config, timeframe)

            # 2. Create Temporal Split Config
            split_config = create_temporal_split_config_for_pipeline(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                data_start=market_data.index.min(),
                data_end=market_data.index.max(),
                enable_burnin=True,
            )
            tprint_info(
                f"📊 Temporal split: Train={split_config.training.effective_end}, "
                f"Test={split_config.test.effective_end}"
            )

            # 3. Generate Features
            tprint_info("🛠️ Generating Volume Force features...")

            # Check cache if in batch mode
            # Cache key includes normalization window as it affects feature values
            norm_window = config.get("volume_force_normalization_window", 100)
            cache_key = (symbol, exchange, timeframe, norm_window)

            if config.get("is_batch_run", False) and cache_key in self._feature_cache:
                tprint_info("Using cached features for batch run")
                feature_df = self._feature_cache[cache_key].copy()
            else:
                # Core Volume Force Features
                force_df = generate_volume_force_features(market_data, config)

                # Shared Liquidity Features (for context)
                liquidity_df = generate_liquidity_regime_features(market_data, config)

                feature_df = pd.concat([force_df, liquidity_df], axis=1)

                # Remove duplicates if any
                feature_df = feature_df.loc[:, ~feature_df.columns.duplicated()]

                # Cache if in batch mode
                if config.get("is_batch_run", False):
                    self._feature_cache[cache_key] = feature_df.copy()

            # Save features artifact (skip in batch mode to save disk I/O unless needed)
            if not config.get("is_batch_run", False):
                feature_df_reset = feature_df.reset_index().rename(columns={feature_df.index.name or "index": "timestamp"})
                features_path = self._save_artifact(
                    data=feature_df_reset,
                    artifact_name="ml_volume_force_features",
                    artifact_type="data",
                    data_category="features",
                    metadata={"source": market_source}
                )
                artifacts.append(features_path)

            # 4. Prepare Training Data
            lookahead = int(config.get("volume_force_lookahead", 12))  # 12 * 15m = 3h
            thresh = float(config.get("volume_force_target_threshold_atr", 1.5))

            # Load meta-label labeled_data artifact and use binary_label as target
            labeled_data = None
            try:
                artifact_name = f"labeled_data_{symbol}_{timeframe}"
                labeled_data = self._get_artifact(artifact_name, "data")
            except Exception:
                try:
                    labeled_data = self._get_artifact("labeled_data", "data")
                except Exception:
                    labeled_data = None

            if labeled_data is None:
                raise RuntimeError("Meta-label labeled_data artifact not available for volume force step")

            if not isinstance(labeled_data, pd.DataFrame):
                labeled_data = pd.DataFrame(labeled_data)

            if labeled_data.index.duplicated().any():
                labeled_data = labeled_data[~labeled_data.index.duplicated(keep="first")]

            labeled_data = labeled_data.reindex(feature_df.index)

            if "binary_label" not in labeled_data.columns:
                raise RuntimeError("binary_label column not found in labeled_data for volume force step")

            y_all = labeled_data["binary_label"].astype(float)

            # Require a minimum number of non-null samples and variation
            min_target_samples = int(config.get("volume_force_min_target_samples", 200))
            non_null = int(y_all.notna().sum())
            unique_vals = int(y_all.nunique(dropna=True))
            if non_null < min_target_samples or unique_vals <= 1:
                raise RuntimeError(
                    f"Insufficient variation in binary_label for volume force step: "
                    f"n_non_null={non_null}, n_unique={unique_vals}"
                )

            # Align target to feature index and drop NaNs / NaN features
            y = y_all.reindex(feature_df.index)
            valid_mask = y.notna() & feature_df.notna().all(axis=1)
            X = feature_df.loc[valid_mask]
            y = y.loc[valid_mask].astype(int)

            numeric_cols = X.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                raise ValueError("No numeric features available for volume force model")
            X = X[numeric_cols]

            class_counts = y.value_counts().to_dict()
            metrics["class_distribution"] = {str(k): int(v) for k, v in class_counts.items()}
            tprint_info(f"🎯 Target class distribution: {class_counts}")

            # 5. Train Model
            model_id = f"{symbol}_{exchange}_{timeframe}_volume_force"

            n_samples = len(X)
            min_samples_for_training = 50
            if n_samples > 0:
                min_samples_for_training = max(50, min(500, n_samples // 3))

            training_config = XGBTrainingConfig(
                model_id=model_id,
                retrain_interval_days=21,
                hpo_interval_days=30,
                burnin_pct=1/12,
                min_samples_for_training=min_samples_for_training,
                n_estimators=800,
                early_stopping_rounds=20,
                tree_method="hist",
                # Binary classification on meta-label binary_label
                objective="binary:logistic",
                # Stronger regularization and smoother probabilities for noisy labels
                learning_rate=0.03,
                min_child_weight=15.0,
                gamma=1.5,
                reg_lambda=3.0,
                reg_alpha=0.5,
                subsample=0.8,
                colsample_bytree=0.8,
            )

            trainer = StandardizedXGBTrainer(model_id=model_id, config=training_config)

            # Time-decay sample weights to emphasize recent regimes
            sample_weights = create_sample_weights(X.index)

            tprint_info("🧠 Training XGBoost (Volume Force)...")
            results = trainer.train_and_predict(
                X=X,
                y=y,
                data_start=X.index.min(),
                data_end=X.index.max(),
                sample_weight=sample_weights,
                eval_metric="mlogloss",
                verbose=True,
            )

            # 6. Process Predictions (OOF)
            # For binary meta-label: scalar in [0,1] is P(binary_label=1 | features)
            oof_preds = results.oof_predictions

            if "probability" in oof_preds.columns and not oof_preds.empty:
                prob = oof_preds["probability"].astype(float).clip(0.0, 1.0)
                oof_preds["scalar_pred"] = prob

                # Save predictions artifact (scalar prediction aligned to timestamp)
                preds_to_save = oof_preds[["scalar_pred"]].reset_index().rename(
                    columns={oof_preds.index.name or "index": "timestamp", "scalar_pred": "predicted"}
                )
                preds_path = self._save_artifact(
                    data=preds_to_save,
                    artifact_name="ml_volume_force_predictions",
                    artifact_type="data",
                    data_category="predictions",
                    metadata={"lookahead": lookahead, "threshold_atr": thresh, "output_type": "scalar_0_1"}
                )
                artifacts.append(preds_path)

                # Binary metrics
                y_true = y.loc[oof_preds.index].astype(int)
                pred_class = (prob >= 0.5).astype(int)
                acc = (pred_class == y_true).mean()
                metrics["accuracy"] = float(acc)
                metrics["scalar_pred_mean"] = float(prob.mean())
                metrics["scalar_pred_std"] = float(prob.std())

                try:
                    ll = log_loss(y_true, prob, labels=[0, 1])
                    metrics["oof_log_loss"] = float(ll)
                except Exception as e:
                    tprint_warning(f"Log loss calculation failed: {e}")
                    metrics["oof_log_loss"] = float("inf")

                # AUC / PR-AUC / Brier
                try:
                    metrics["auc_roc"] = float(roc_auc_score(y_true, prob))
                except Exception as e:
                    tprint_warning(f"ROC AUC calculation failed: {e}")

                try:
                    metrics["pr_auc"] = float(average_precision_score(y_true, prob))
                except Exception as e:
                    tprint_warning(f"PR AUC calculation failed: {e}")

                try:
                    metrics["brier_score"] = float(brier_score_loss(y_true, prob))
                except Exception as e:
                    tprint_warning(f"Brier score calculation failed: {e}")

                # Record OOF coverage for downstream analysis and sweep ranking
                metrics["n_oof_samples"] = int(len(oof_preds))
                oof_start = oof_preds.index.min()
                oof_end = oof_preds.index.max()
                metrics["oof_start"] = oof_start
                metrics["oof_end"] = oof_end

                tprint_success(
                    f"✅ Training complete. OOF Accuracy: {acc:.4f}, LogLoss: {metrics.get('oof_log_loss', 'N/A'):.4f}"
                )
                tprint_info(
                    f"   Scalar prediction stats: Mean={prob.mean():.3f}, Std={prob.std():.3f}"
                )
            else:
                tprint_warning("Could not map predictions to scalar (missing 'probability' column).")

            # Save Model (if any)
            model_path = None
            if results.models:
                model_path = self._save_artifact(
                    data=results.models[-1],
                    artifact_name="ml_volume_force_model",
                    artifact_type="model",
                    data_category="models",
                    metadata={"n_features": len(X.columns)}
                )
                artifacts.append(model_path)

            # Save metrics artifact (only for non-batch runs)
            if not config.get("is_batch_run", False) and metrics.get("oof_log_loss") is not None:
                if not oof_preds.empty:
                    oof_start = metrics.get("oof_start")
                    oof_end = metrics.get("oof_end")
                    metrics_artifact = {
                        "oof_log_loss": float(metrics.get("oof_log_loss", float("inf"))),
                        "accuracy": float(metrics.get("accuracy", 0.0)),
                        "auc_roc": float(metrics.get("auc_roc", float("nan"))),
                        "pr_auc": float(metrics.get("pr_auc", float("nan"))),
                        "brier_score": float(metrics.get("brier_score", float("nan"))),
                        "class_distribution": metrics.get("class_distribution", {}),
                        "n_samples_total": int(metrics.get("n_samples", len(X))),
                        "n_oof_samples": int(metrics.get("n_oof_samples", len(oof_preds))),
                        "oof_start": oof_start.isoformat() if hasattr(oof_start, "isoformat") else str(oof_start),
                        "oof_end": oof_end.isoformat() if hasattr(oof_end, "isoformat") else str(oof_end),
                        "lookahead": int(lookahead),
                        "threshold_atr": float(thresh),
                    }
                    metrics_path = self._save_artifact(
                        data=metrics_artifact,
                        artifact_name="ml_volume_force_metrics",
                        artifact_type="metadata",
                        data_category="metrics",
                    )
                    artifacts.append(metrics_path)

            metrics["n_samples"] = len(X)
            metrics["execution_time"] = time.time() - start_time

            return {
                "success": True,
                "artifacts": artifacts,
                "metrics": metrics,
                "model_path": model_path if results.models else None,
                "execution_time": metrics["execution_time"]
            }

        except Exception as exc:
            tprint_error(f"❌ {self.step_name} failed: {exc}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(exc),
                "execution_time": time.time() - start_time
            }

    def _load_market_data_with_cache(self, config: Dict[str, Any], timeframe: str) -> Tuple[pd.DataFrame, str]:
        """Load market data with caching."""
        symbol = config.get("symbol")
        exchange = config.get("exchange")
        exec_mode = str(config.get("execution_mode", "")).lower()

        cache_key = (symbol, exchange, timeframe, exec_mode)

        if (self._cached_market_data is not None and
            self._cached_market_cache_key == cache_key):
            return self._cached_market_data.copy(), self._cached_market_source

        market_data, market_source = self.load_market_data_or_fail(
            {**config, "timeframe": timeframe},
            pipeline_state={},
            allow_config_override=True,
            light_mode_filter=False,
            skip_artifacts=True,
        )

        # Ensure DateTimeIndex and numeric types
        if not isinstance(market_data.index, pd.DatetimeIndex):
            market_data.index = pd.to_datetime(market_data.index)

        for col in ["open", "high", "low", "close", "volume"]:
            if col in market_data.columns:
                market_data[col] = pd.to_numeric(market_data[col], errors='coerce')

        self._cached_market_data = market_data.copy()
        self._cached_market_source = market_source
        self._cached_market_cache_key = cache_key

        return market_data, market_source
