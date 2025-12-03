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
from src.feature_generation.categories.volatility import generate_volatility_features # For ATR if needed

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
                    "oof_log_loss": metrics.get("avg_log_loss", float("inf")),
                    "oof_accuracy": metrics.get("avg_accuracy", 0.0),

                    # Individual Model Metrics
                    "breakout_log_loss": metrics.get("breakout_log_loss", float("inf")),
                    "volatility_log_loss": metrics.get("volatility_log_loss", float("inf")),
                    "trend_log_loss": metrics.get("trend_log_loss", float("inf")),

                    "n_samples": metrics.get("n_samples", 0),
                    "n_oof_samples": metrics.get("n_oof_samples", 0),
                    "oof_start": metrics.get("oof_start"),
                    "oof_end": metrics.get("oof_end"),
                }

                run_metrics.update({
                    f"config_{k}": v for k, v in config.items()
                    if k.startswith("volume_force_")
                })

                results.append(run_metrics)

                if result.get("success", False):
                    tprint_success(
                        f"✅ Config {i+1} done. Avg Loss: {run_metrics['oof_log_loss']:.4f}"
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

            sample_weights = create_sample_weights(X.index)

            for target_name in target_names:
                tprint_info(f"🧠 Training {target_name.capitalize()} Model...")

                model_id = f"{symbol}_{exchange}_{timeframe}_volume_force_{target_name}"
                y_target = y[f"target_{target_name}"].astype(int)

                # Check target balance
                pos_ratio = y_target.mean()
                tprint_info(f"   Target Balance ({target_name}): {pos_ratio:.1%}")

                if pos_ratio < 0.01 or pos_ratio > 0.99:
                    tprint_warning(f"⚠️ Extreme class imbalance for {target_name}, skipping or using dummy")
                    # Handle extreme imbalance if necessary, but XGB usually handles it well enough or we subsample
                    # For now proceeding, but logging warning.

                training_config = XGBTrainingConfig(
                    model_id=model_id,
                    retrain_interval_days=21,
                    hpo_interval_days=30,
                    burnin_pct=1/12,
                    min_samples_for_training=200,
                    n_estimators=800,
                    early_stopping_rounds=20,
                    tree_method="hist",
                    objective="binary:logistic",
                    learning_rate=0.03,
                    min_child_weight=10.0,
                    gamma=1.5,
                    reg_lambda=3.0,
                    subsample=0.8,
                    colsample_bytree=0.8,
                )

                trainer = StandardizedXGBTrainer(model_id=model_id, config=training_config)

                train_result = trainer.train_and_predict(
                    X=X,
                    y=y_target,
                    data_start=X.index.min(),
                    data_end=X.index.max(),
                    sample_weight=sample_weights,
                    eval_metric="logloss",
                    verbose=False,
                )

                model_results[target_name] = train_result
                trained_models[target_name] = train_result.models[-1] if train_result.models else None

                # Extract OOF probas
                oof_preds = train_result.oof_predictions
                if "probability" in oof_preds.columns:
                    # Align OOF preds to common index (X.index)
                    # Note: train_result.oof_predictions should already be indexed by X.index subset
                    # We merge into our main predictions dataframe
                    predictions[f"vol_force_{target_name}"] = oof_preds["probability"]
                else:
                    predictions[f"vol_force_{target_name}"] = np.nan

                # Metrics
                if not oof_preds.empty:
                    y_true = y_target.loc[oof_preds.index]
                    y_prob = oof_preds["probability"]
                    try:
                        ll = log_loss(y_true, y_prob)
                        acc = (y_true == (y_prob >= 0.5)).mean()
                        metrics[f"{target_name}_log_loss"] = float(ll)
                        metrics[f"{target_name}_accuracy"] = float(acc)
                    except Exception:
                        metrics[f"{target_name}_log_loss"] = float("inf")

            # 5. Aggregate Results
            # Filter rows where we have predictions for all (intersection of valid OOFs)
            predictions = predictions.dropna()

            if predictions.empty:
                tprint_warning("⚠️ No common OOF predictions generated.")
                metrics["avg_log_loss"] = float("inf")
            else:
                avg_ll = np.mean([
                    metrics.get(f"{t}_log_loss", float("inf"))
                    for t in target_names
                ])
                avg_acc = np.mean([
                    metrics.get(f"{t}_accuracy", 0.0)
                    for t in target_names
                ])
                metrics["avg_log_loss"] = float(avg_ll)
                metrics["avg_accuracy"] = float(avg_acc)
                metrics["n_oof_samples"] = len(predictions)
                metrics["oof_start"] = predictions.index.min()
                metrics["oof_end"] = predictions.index.max()

                # Save Predictions Artifact
                preds_to_save = predictions.reset_index().rename(columns={predictions.index.name or "index": "timestamp"})
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

                tprint_success(
                    f"✅ Training Complete. Avg OOF LogLoss: {avg_ll:.4f}, Acc: {avg_acc:.4f}"
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
        vol_percentile = 75
        beta = 0.75  # Trend start threshold
        gamma = 0.5  # Trend continuation threshold

        targets = pd.DataFrame(index=df.index)

        # --- Target A: Breakout ---
        # High/Low over past L
        past_high = df["high"].rolling(L).max()
        past_low = df["low"].rolling(L).min()

        # Future High/Low over next H
        # rolling(H) at t includes t. We want future: t+1 to t+H.
        # Shift back by H?
        # rolling(H) at t is [t-H+1, t].
        # We want max([t+1, ..., t+H]).
        # Shift df['high'] by -H, then rolling(H) -> window ends at t+H, contains [t+1, t+H].
        future_high = df["high"].shift(-H).rolling(window=H).max()
        future_low = df["low"].shift(-H).rolling(window=H).min()

        # ATR for threshold
        # Simple ATR approximation: High-Low rolling mean
        # Or True Range.
        tr = np.maximum(df["high"] - df["low"],
                        np.abs(df["high"] - df["close"].shift(1)))
        atr = tr.rolling(14).mean()
        threshold_val = atr * atr_thresh_mult

        # Breakout Condition
        breakout_up = future_high > (past_high + threshold_val)
        breakout_down = future_low < (past_low - threshold_val)
        targets["target_breakout"] = (breakout_up | breakout_down).astype(int)

        # --- Target B: Volatility (High Regime) ---
        # Future Realized Volatility (RV) over H
        # RV = StdDev of returns * sqrt(H) ? Or just StdDev over H bars.
        returns = df["close"].pct_change()
        # rv_future at t = std(returns[t+1...t+H])
        rv_series = returns.rolling(H).std() # at t, is std(t-H+1...t)
        rv_future = rv_series.shift(-H) # at t, is std(t+1...t+H)

        # Past RV Distribution (Window W)
        # We compare rv_future[t] to distribution of rv_series[t-W ... t]
        # rolling(W) on rv_series gives us the window of past H-bar volatilities.
        rv_threshold = rv_series.rolling(W).quantile(vol_percentile / 100.0)

        targets["target_volatility"] = (rv_future > rv_threshold).astype(int)

        # --- Target C: Trend Persistence ---
        # r_past = (close[t] - close[t-H]) / close[t-H]
        # Note: standard pct_change(H) is (c[t] - c[t-H])/c[t-H]
        r_past = df["close"].pct_change(H)

        # r_future = (close[t+H] - close[t]) / close[t]
        r_future = df["close"].pct_change(H).shift(-H)

        # Sigma Ref: Rolling std of 1-bar returns over K bars, scaled to H-bar horizon
        # sigma_1bar = returns.rolling(K).std()
        # sigma_ref = sigma_1bar * np.sqrt(H)
        sigma_1bar = returns.rolling(K).std()
        sigma_ref = sigma_1bar * np.sqrt(H)

        min_return = beta * sigma_ref
        min_return_future = gamma * sigma_ref

        # Identify Current Trend
        # trend_sign = sign(r_past) if abs(r_past) >= min_return else 0
        trend_sign = np.sign(r_past)
        is_trend = r_past.abs() >= min_return
        trend_sign = np.where(is_trend, trend_sign, 0)

        # Persistence Condition
        # 1 if trend exists AND future direction matches AND future magnitude sufficient
        persistence = (
            (trend_sign != 0) &
            (np.sign(r_future) == trend_sign) &
            (r_future.abs() >= min_return_future)
        )
        targets["target_trend"] = persistence.astype(int)

        return targets

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
