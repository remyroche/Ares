"""
ML Volume Force Step

This step constructs Volume Force/Impulse features and trains three separate XGBoost models
to predict market regime targets: Breakout, Volatility, and Trend Persistence.

Targets:
1. Breakout Probability (Classifier): Will price break out of recent range within horizon?
2. High Volatility Probability (Classifier): Will future realized volatility be in the top quartile?
3. Trend Persistence (Classifier): Will the current trend continue with sufficient magnitude?

Outputs:
- Artifact `ml_volume_force_predictions` with columns:
  - vol_force_breakout
  - vol_force_volatility
  - vol_force_trend
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    log_loss,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.isotonic import IsotonicRegression

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
    """Pipeline step for Volume Force multi-task prediction."""

    def __init__(self, step_name: str = "ml_volume_force_step"):
        super().__init__(step_name, use_versioned_artifacts=True)
        self.logger = logger.getChild("MLVolumeForceStep") if hasattr(logger, "getChild") else logger
        self._cached_market_data = None
        self._cached_market_source = None
        self._cached_market_cache_key = None
        self._feature_cache = {}
        tprint(f"✅ Initialized {step_name} step (Multi-Target)", "SUCCESS")

    async def run_config_batch(
        self, configs: List[Dict[str, Any]], symbol: str, exchange: str
    ) -> List[Dict[str, Any]]:
        """Run a batch of configurations and collect results."""
        results = []
        total_configs = len(configs)

        for i, config in enumerate(configs):
            config["symbol"] = symbol
            config["exchange"] = exchange
            config["execution_mode"] = config.get("execution_mode", "light")
            config["is_batch_run"] = True

            tprint_info(
                f"🚀 Running config {i+1}/{total_configs}: {self.get_config_signature(config)}"
            )

            try:
                start_time = time.time()
                result = await self.execute(config)
                execution_time = time.time() - start_time

                metrics = result.get("metrics", {})

                run_metrics = {
                    "config_id": i + 1,
                    "config_signature": self.get_config_signature(config),
                    "execution_time": execution_time,
                    "success": result.get("success", False),
                    "error": result.get("error", ""),

                    # Aggregated Metrics (Mean of 3 models)
                    "oof_log_loss": metrics.get("avg_log_loss", float("inf")),  # Still valid for breakout?
                    "oof_accuracy": metrics.get("avg_accuracy", 0.0),

                    # Individual Model Metrics
                    "breakout_log_loss": metrics.get("breakout_log_loss", float("inf")),
                    "volatility_rmse": metrics.get("volatility_rmse", float("inf")),
                    "trend_rmse": metrics.get("trend_rmse", float("inf")),
                    "trend_ic": metrics.get("trend_ic", 0.0),

                    "n_samples": metrics.get("n_samples", 0),
                    "n_oof_samples": metrics.get("n_oof_samples", 0),
                    "oof_start": metrics.get("oof_start"),
                    "oof_end": metrics.get("oof_end"),
                }

                run_metrics.update({
                    f"config_{k}": v for k, v in config.items()
                    if k.startswith("volume_force_")
                })

                # Include detailed per-target metrics to make sweep CSV richer
                for key, value in metrics.items():
                    if key not in run_metrics and (
                        key.startswith("breakout_")
                        or key.startswith("volatility_")
                        or key.startswith("trend_")
                    ):
                        run_metrics[key] = value

                results.append(run_metrics)

                if result.get("success", False):
                    # Log diverse metrics for clarity
                    tprint_success(
                        f"✅ Config {i+1} done. "
                        f"Breakout Loss: {run_metrics.get('breakout_log_loss', 99.9):.4f}, "
                        f"Trend IC: {run_metrics.get('trend_ic', 0.0):.4f}, "
                        f"Vol RMSE: {run_metrics.get('volatility_rmse', 99.9):.4f}"
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
        successful = df[df["success"] == True].copy()

        if successful.empty:
            return df, {"best_config": None, "analysis": "no_successful_runs"}

        # Sort by Average Log Loss
        successful = successful.sort_values("oof_log_loss", ascending=True)

        best_run = successful.iloc[0].to_dict()
        best_config = {}
        for k, v in best_run.items():
            if k.startswith("config_"):
                best_config[k.replace("config_", "")] = v

        analysis = {
            "best_config": best_config,
            "best_log_loss": best_run["oof_log_loss"],
            "total_runs": len(results),
            "successful_runs": len(successful),
        }

        return df, analysis

    def get_config_signature(self, config: Dict[str, Any]) -> str:
        keys = [
            "volume_force_target_threshold_atr",
            "volume_force_lookahead",
            "volume_force_normalization_window",
            "volume_force_volatility_percentile",
            "volume_force_trend_beta",
            "volume_force_xgb_max_depth"
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
                f"🚀 Starting {self.step_name} (Multi-Target) for {symbol} on {exchange}"
            )

            # 1. Load Market Data
            market_data, market_source = self._load_market_data_with_cache(config, timeframe)

            # 2. Generate Features
            tprint_info("🛠️ Generating Volume Force features...")
            norm_window = config.get("volume_force_normalization_window", 100)
            cache_key = (symbol, exchange, timeframe, norm_window)

            if config.get("is_batch_run", False) and cache_key in self._feature_cache:
                feature_df = self._feature_cache[cache_key].copy()
            else:
                force_df = generate_volume_force_features(market_data, config)
                liquidity_df = generate_liquidity_regime_features(market_data, config)
                feature_df = pd.concat([force_df, liquidity_df], axis=1)
                feature_df = feature_df.loc[:, ~feature_df.columns.duplicated()]

                if config.get("is_batch_run", False):
                    self._feature_cache[cache_key] = feature_df.copy()

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

            # 3. Generate Targets
            tprint_info("🎯 Generating Targets (Breakout, Volatility, Trend)...")
            targets_df = self._generate_targets(market_data, config)

            # Align features and targets
            # Ensure index alignment
            common_index = feature_df.index.intersection(targets_df.index)
            X = feature_df.loc[common_index]
            y = targets_df.loc[common_index]

            # Drop NaN rows where features or targets are missing
            valid_mask = X.notna().all(axis=1) & y.notna().all(axis=1)
            X = X.loc[valid_mask]
            y = y.loc[valid_mask]

            numeric_cols = X.select_dtypes(include=[np.number]).columns
            X = X[numeric_cols]

            if len(X) < 200:
                raise RuntimeError(f"Insufficient valid samples: {len(X)} < 200")

            tprint_info(f"📊 Training Data: {len(X)} samples")

            # 4. Train 3 Models
            model_results = {}
            predictions = pd.DataFrame(index=X.index)
            trained_models = {}

            target_names = ["breakout", "volatility", "trend"]

            base_sample_weights = create_sample_weights(X.index).astype(float)

            for target_name in target_names:
                tprint_info(f"🧠 Training {target_name.capitalize()} Model...")

                model_id = f"{symbol}_{exchange}_{timeframe}_volume_force_{target_name}"
                y_target = y[f"target_{target_name}"]

                # Determine objective and metric based on target type
                if target_name == "breakout":
                    objective = "binary:logistic"
                    eval_metric = "logloss"
                    # Breakout is binary, ensure int
                    y_target = y_target.astype(int)
                    # Use balancing weights for classification
                    pos_ratio = y_target.mean()
                    tprint_info(f"   Target Balance ({target_name}): {pos_ratio:.1%}")

                    target_weights = base_sample_weights.copy()
                    eps = 1e-6
                    if 0.0 < float(pos_ratio) < 1.0:
                        neg_ratio = 1.0 - float(pos_ratio)
                        pos_w = 0.5 / max(float(pos_ratio), eps)
                        neg_w = 0.5 / max(float(neg_ratio), eps)
                        class_weights = np.where(y_target.values == 1, pos_w, neg_w)
                        target_weights = target_weights * class_weights

                    # Optionally tilt weights toward breakouts with larger future moves.
                    # This encourages the classifier to focus more on economically
                    # meaningful events without changing the target definition.
                    move_weight_scale = float(config.get("volume_force_breakout_move_weight_scale", 0.5))
                    move_weight_cap = float(config.get("volume_force_breakout_move_weight_cap", 3.0))
                    if "future_return_H" in y.columns and move_weight_scale > 0.0:
                        try:
                            fwd = y["future_return_H"].loc[y_target.index].astype(float)
                            fwd_abs = fwd.abs().replace([np.inf, -np.inf], np.nan).fillna(0.0)

                            positive_mask = fwd_abs > 0
                            ref_scale = 0.0
                            if positive_mask.any():
                                ref_scale = float(fwd_abs[positive_mask].quantile(0.75))
                            if not np.isfinite(ref_scale) or ref_scale <= 0.0:
                                ref_scale = float(fwd_abs.max())
                            if not np.isfinite(ref_scale) or ref_scale <= 0.0:
                                ref_scale = 0.01

                            move_intensity = (fwd_abs / ref_scale).clip(0.0, move_weight_cap)
                            move_weights = 1.0 + move_weight_scale * move_intensity.to_numpy()
                            # Only upweight positive breakout examples; keep negatives at 1.0
                            move_weights = np.where(y_target.values == 1, move_weights, 1.0)
                            target_weights = target_weights * move_weights
                        except Exception as e:
                            tprint_warning(f"Failed to apply breakout move-based weighting: {e}")

                else:
                    # Volatility and Trend are Regression
                    objective = "reg:squarederror"
                    eval_metric = "rmse"
                    y_target = y_target.astype(float)
                    # For regression, use sample weights (recency) but no class balancing
                    target_weights = base_sample_weights.copy()
                    tprint_info(f"   Target Mean ({target_name}): {y_target.mean():.4f}, Std: {y_target.std():.4f}")

                # Hyperparameters from config (sweeping support)
                xgb_lr = float(config.get("volume_force_xgb_learning_rate", 0.03))
                xgb_depth = int(config.get("volume_force_xgb_max_depth", 6))
                xgb_n_estimators = int(config.get("volume_force_xgb_n_estimators", 800))
                xgb_min_child_weight = float(config.get("volume_force_xgb_min_child_weight", 10.0))
                xgb_gamma = float(config.get("volume_force_xgb_gamma", 1.5))
                xgb_reg_lambda = float(config.get("volume_force_xgb_reg_lambda", 3.0))
                xgb_subsample = float(config.get("volume_force_xgb_subsample", 0.8))
                xgb_colsample_bytree = float(config.get("volume_force_xgb_colsample_bytree", 0.8))

                training_config = XGBTrainingConfig(
                    model_id=model_id,
                    retrain_interval_days=21,
                    hpo_interval_days=30,
                    burnin_pct=1/12,
                    min_samples_for_training=200,
                    n_estimators=xgb_n_estimators,
                    early_stopping_rounds=20,
                    tree_method="hist",
                    objective=objective,
                    learning_rate=xgb_lr,
                    max_depth=xgb_depth,
                    min_child_weight=xgb_min_child_weight,
                    gamma=xgb_gamma,
                    reg_lambda=xgb_reg_lambda,
                    subsample=xgb_subsample,
                    colsample_bytree=xgb_colsample_bytree,
                )

                trainer = StandardizedXGBTrainer(model_id=model_id, config=training_config)

                train_result = trainer.train_and_predict(
                    X=X,
                    y=y_target,
                    data_start=X.index.min(),
                    data_end=X.index.max(),
                    sample_weight=target_weights,
                    eval_metric=eval_metric,
                    verbose=False,
                )

                model_results[target_name] = train_result
                trained_models[target_name] = train_result.models[-1] if train_result.models else None

                # Extract OOF preds
                oof_preds = train_result.oof_predictions
                # Column name depends on objective: 'probability' for binary, 'prediction' for reg
                pred_col = "probability" if "probability" in oof_preds.columns else "prediction"

                if pred_col in oof_preds.columns:
                    prob_series = oof_preds[pred_col]
                    if not prob_series.index.is_unique:
                        prob_series = prob_series[~prob_series.index.duplicated(keep="last")]
                    aligned = prob_series.reindex(predictions.index)
                    predictions[f"vol_force_{target_name}"] = aligned
                else:
                    predictions[f"vol_force_{target_name}"] = np.nan

                # Metrics
                if not oof_preds.empty:
                    y_true = y_target.loc[oof_preds.index]
                    y_pred = oof_preds[pred_col]

                    try:
                        if target_name == "breakout":
                            # Classification Metrics
                            ll = log_loss(y_true, y_pred)
                            acc = (y_true == (y_pred >= 0.5)).mean()
                            metrics[f"{target_name}_log_loss"] = float(ll)
                            metrics[f"{target_name}_accuracy"] = float(acc)

                            try:
                                roc = roc_auc_score(y_true, y_pred)
                                pr = average_precision_score(y_true, y_pred)
                                brier = brier_score_loss(y_true, y_pred)
                                metrics[f"{target_name}_roc_auc"] = float(roc)
                                metrics[f"{target_name}_pr_auc"] = float(pr)
                                metrics[f"{target_name}_brier_score"] = float(brier)
                            except Exception:
                                pass

                            # Lift stats for Breakout
                            base_rate = float(y_true.mean()) if len(y_true) > 0 else 0.0
                            if len(y_pred) > 0:
                                for q, label in ((0.95, "top5"), (0.90, "top10")):
                                    q_thresh = float(y_pred.quantile(q))
                                    mask = y_pred >= q_thresh
                                    key_prefix = f"{target_name}_{label}"
                                    if mask.any():
                                        precision = float(y_true[mask].mean())
                                        lift = float(precision / base_rate) if base_rate > 0 else 0.0
                                        metrics[f"{key_prefix}_lift"] = lift
                                    else:
                                        metrics[f"{key_prefix}_lift"] = 0.0

                        else:
                            # Regression Metrics
                            mse = mean_squared_error(y_true, y_pred)
                            rmse = np.sqrt(mse)
                            mae = mean_absolute_error(y_true, y_pred)
                            r2 = r2_score(y_true, y_pred)

                            metrics[f"{target_name}_rmse"] = float(rmse)
                            metrics[f"{target_name}_mae"] = float(mae)
                            metrics[f"{target_name}_r2"] = float(r2)

                            # Information Coefficient (IC) - Correlation between pred and target
                            ic = np.corrcoef(y_pred, y_true)[0, 1]
                            metrics[f"{target_name}_ic"] = float(ic)

                    except Exception as e:
                        tprint_warning(f"Error calculating metrics for {target_name}: {e}")
                        if target_name == "breakout":
                            metrics[f"{target_name}_log_loss"] = float("inf")
                        else:
                            metrics[f"{target_name}_rmse"] = float("inf")

            # 5. Aggregate Results
            # Filter rows where we have predictions for all (intersection of valid OOFs)
            predictions = predictions.dropna()

            if predictions.empty:
                tprint_warning("⚠️ No common OOF predictions generated.")
                metrics["avg_log_loss"] = float("inf")
            else:
                # Calculate average 'error' metric (Loss for classification, RMSE for regression)
                # Normalizing them is hard, so we just average the available "primary" metrics for sorting
                # But typically log_loss and RMSE are different scales.
                # For backward compatibility, we set avg_log_loss to breakout_log_loss
                # or a mix.
                metrics["avg_log_loss"] = metrics.get("breakout_log_loss", float("inf"))
                metrics["avg_accuracy"] = metrics.get("breakout_accuracy", 0.0)
                metrics["n_oof_samples"] = len(predictions)
                metrics["oof_start"] = predictions.index.min()
                metrics["oof_end"] = predictions.index.max()

                # Attach targets and simple forward return to predictions for downstream analysis
                preds_with_targets = predictions.copy()
                for tname in target_names:
                    col_name = f"target_{tname}"
                    if col_name in y.columns:
                        preds_with_targets[col_name] = y.loc[preds_with_targets.index, col_name].astype(float)
                if "future_return_H" in y.columns:
                    preds_with_targets["future_return_H"] = y.loc[preds_with_targets.index, "future_return_H"].astype(float)

                # Calibrate breakout probabilities into a 0-1 scalar score when possible.
                if "vol_force_breakout" in preds_with_targets.columns and "target_breakout" in preds_with_targets.columns:
                    raw_prob = preds_with_targets["vol_force_breakout"].astype(float)
                    y_break = preds_with_targets["target_breakout"].astype(float)

                    raw_vals = raw_prob.values
                    y_vals = y_break.values
                    mask = (
                        np.isfinite(raw_vals)
                        & np.isfinite(y_vals)
                    )

                    min_cal_samples = int(config.get("volume_force_min_calibration_samples", 200))
                    if mask.sum() >= max(50, min_cal_samples) and np.unique(y_vals[mask]).size >= 2:
                        try:
                            ir = IsotonicRegression(y_min=0.0, y_max=1.0)
                            ir.fit(raw_vals[mask], y_vals[mask])

                            prob_cal = np.full_like(raw_vals, np.nan, dtype=float)
                            valid_mask = np.isfinite(raw_vals)
                            if valid_mask.any():
                                prob_cal[valid_mask] = ir.transform(raw_vals[valid_mask])

                            prob_series = pd.Series(prob_cal, index=raw_prob.index).clip(0.0, 1.0)
                            preds_with_targets["vol_force_breakout_score"] = prob_series
                            metrics["breakout_calibration_method"] = "isotonic_oof"
                        except Exception:
                            preds_with_targets["vol_force_breakout_score"] = raw_prob.clip(0.0, 1.0)
                            metrics["breakout_calibration_method"] = "identity_fallback"
                    else:
                        preds_with_targets["vol_force_breakout_score"] = raw_prob.clip(0.0, 1.0)
                        metrics["breakout_calibration_method"] = "identity_insufficient_samples"

                breakout_metrics = self._compute_breakout_trading_effectiveness(preds_with_targets, config)
                for k, v in breakout_metrics.items():
                    if k not in metrics:
                        metrics[k] = v

                # Save Predictions Artifact
                preds_to_save = preds_with_targets.reset_index().rename(columns={preds_with_targets.index.name or "index": "timestamp"})
                preds_path = self._save_artifact(
                    data=preds_to_save,
                    artifact_name="ml_volume_force_predictions",
                    artifact_type="data",
                    data_category="predictions",
                    metadata={
                        "lookahead": config.get("volume_force_lookahead"),
                        "columns": list(predictions.columns)
                    }
                )
                artifacts.append(preds_path)

                avg_ll = float(metrics.get("avg_log_loss", float("inf")) or float("inf"))
                avg_acc = float(metrics.get("avg_accuracy", 0.0) or 0.0)

                tprint_success(
                    f" Training Complete. Avg OOF LogLoss: {avg_ll:.4f}, Acc: {avg_acc:.4f}"
                )

            # Save Models (Dict of models is not directly supported by save_artifact for 'model' type usually,
            # but we can save picklable object)
            # Or save individual models. For now, skipping saving models to disk to save space/time unless needed for inference.
            # In production, we'd save each trainer's model separately.

            metrics["n_samples"] = len(X)
            metrics["execution_time"] = time.time() - start_time

            return {
                "success": True,
                "artifacts": artifacts,
                "metrics": metrics,
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

    def _generate_targets(self, market_data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Generate classification targets for Breakout, Volatility, and Trend."""
        df = market_data.copy()

        # Params
        H = int(config.get("volume_force_lookahead", 12))  # Horizon
        L = 20  # Breakout Lookback
        W = 96  # Volatility Distribution Window
        K = 96  # Trend Reference Window

        atr_thresh_mult = float(config.get("volume_force_target_threshold_atr", 1.0))
        vol_percentile = float(config.get("volume_force_volatility_percentile", 75))
        beta = float(config.get("volume_force_trend_beta", 0.75))
        gamma = float(config.get("volume_force_trend_gamma", 0.5))

        targets = pd.DataFrame(index=df.index)

        # --- Target A: Breakout (Refactored: Close-based validation) ---
        # High/Low over past L
        past_high = df["high"].rolling(L).max()
        past_low = df["low"].rolling(L).min()

        # ATR for threshold
        tr = np.maximum(df["high"] - df["low"],
                        np.abs(df["high"] - df["close"].shift(1)))
        atr = tr.rolling(14).mean()
        threshold_val = atr * atr_thresh_mult

        # Future Close High/Low (not Wick)
        # Check if the MAXIMUM CLOSE in the next H bars exceeds the level.
        # This confirms the price *stayed* or *closed* above the level, reducing wick fakeouts.
        future_close_max = df["close"].shift(-H).rolling(window=H).max()
        future_close_min = df["close"].shift(-H).rolling(window=H).min()

        # Breakout Condition:
        # Upside: Max future close > (Past High + Threshold)
        # Downside: Min future close < (Past Low - Threshold)
        breakout_up = future_close_max > (past_high + threshold_val)
        breakout_down = future_close_min < (past_low - threshold_val)
        targets["target_breakout"] = (breakout_up | breakout_down).astype(int)

        # --- Target B: Volatility (Regression) ---
        # Target: Future Realized Volatility over horizon H
        returns = df["close"].pct_change()
        # rv_future at t = std(returns[t+1...t+H])
        rv_series = returns.rolling(H).std()
        rv_future = rv_series.shift(-H)

        # Log-transform volatility to make it more Gaussian-like for regression
        # Avoid log(0) with eps
        targets["target_volatility"] = np.log1p(rv_future * 100) # Scaling up before log

        # --- Target C: Trend (Regression) ---
        # Target: Future Return over horizon H (Directional Strength)
        r_future = df["close"].pct_change(H).shift(-H)

        # Clip extreme returns to stabilize regression (e.g., +/- 20%)
        # Though XGBoost is robust, clipping helps avoid outliers driving loss
        targets["target_trend"] = r_future.clip(-0.2, 0.2)

        # Simple H-bar forward return for backtesting/analysis
        targets["future_return_H"] = r_future

        return targets

    def _compute_breakout_trading_effectiveness(
        self,
        preds_with_targets: pd.DataFrame,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}
        score_col = "vol_force_breakout_score" if "vol_force_breakout_score" in preds_with_targets.columns else "vol_force_breakout"
        if score_col not in preds_with_targets.columns:
            return metrics
        if "future_return_H" not in preds_with_targets.columns:
            return metrics

        prob = preds_with_targets[score_col].astype(float)
        fwd = preds_with_targets["future_return_H"].astype(float)

        mask = prob.notna() & fwd.notna() & np.isfinite(fwd.values)
        prob = prob[mask]
        fwd = fwd[mask]
        if len(prob) == 0:
            return metrics

        raw_thresholds = config.get("volume_force_trade_thresholds")
        if isinstance(raw_thresholds, (list, tuple)):
            thresholds = [float(t) for t in raw_thresholds if t is not None]
        elif raw_thresholds is not None:
            thresholds = [float(raw_thresholds)]
        else:
            thresholds = [0.6, 0.7, 0.8]

        raw_quantiles = config.get("volume_force_trade_quantiles")
        if isinstance(raw_quantiles, (list, tuple)):
            quantiles = [float(q) for q in raw_quantiles if q is not None]
        elif raw_quantiles is not None:
            quantiles = [float(raw_quantiles)]
        else:
            # Default to a small set of high-quantile slices, including top-30%
            quantiles = [0.95, 0.9, 0.7]

        def _add_stats(prefix: str, sel: pd.Series) -> None:
            n = int(sel.shape[0])
            total = int(fwd.shape[0])
            coverage = float(n / total) if total > 0 else 0.0
            mean_ret = float(sel.mean()) if n > 0 else 0.0
            std_ret = float(sel.std(ddof=0)) if n > 1 else 0.0
            if n > 0:
                hit_rate = float((sel > 0).mean())
            else:
                hit_rate = 0.0
            if std_ret > 0.0 and n > 1:
                sharpe = float(mean_ret / std_ret * np.sqrt(float(n)))
            else:
                sharpe = 0.0
            metrics[prefix + "_coverage"] = coverage
            metrics[prefix + "_hit_rate"] = hit_rate
            metrics[prefix + "_mean_return"] = mean_ret
            metrics[prefix + "_sharpe"] = sharpe
            metrics[prefix + "_n_trades"] = n

        for thr in thresholds:
            try:
                thr_val = float(thr)
            except (TypeError, ValueError):
                continue
            mask_thr = prob >= thr_val
            if not mask_thr.any():
                continue
            sel = fwd[mask_thr]
            label_val = int(round(thr_val * 100))
            prefix = f"breakout_trade_p{label_val}"
            _add_stats(prefix, sel)

        for q in quantiles:
            try:
                q_val = float(q)
            except (TypeError, ValueError):
                continue
            if not 0.0 < q_val < 1.0:
                continue
            prob_thr = float(prob.quantile(q_val))
            mask_q = prob >= prob_thr
            if not mask_q.any():
                continue
            sel = fwd[mask_q]
            pct = int(round((1.0 - q_val) * 100))
            prefix = f"breakout_trade_top{pct}"
            _add_stats(prefix, sel)

        return metrics

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

        if not isinstance(market_data.index, pd.DatetimeIndex):
            market_data.index = pd.to_datetime(market_data.index)

        for col in ["open", "high", "low", "close", "volume"]:
            if col in market_data.columns:
                market_data[col] = pd.to_numeric(market_data[col], errors='coerce')

        self._cached_market_data = market_data.copy()
        self._cached_market_source = market_source
        self._cached_market_cache_key = cache_key

        return market_data, market_source
