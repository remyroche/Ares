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
from sklearn.metrics import log_loss

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

        # Sort by Log Loss (Lower is better)
        successful = successful.sort_values("oof_log_loss", ascending=True)

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
            norm_window = config.get("volume_force_normalization_window", 500)
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

            # Target: Forward Return
            # We want to predict direction: Down (-1), Neutral (0), Up (1)
            # Thresholding based on volatility (ATR) or fixed percentage?
            # Using fixed percentage for simplicity and stability, or ATR-based.
            # Let's use ATR-normalized return to be robust across regimes.

            if "atr" not in market_data.columns:
                 # Quick ATR calculation if missing
                 high = market_data["high"]
                 low = market_data["low"]
                 close = market_data["close"]
                 tr1 = high - low
                 tr2 = (high - close.shift(1)).abs()
                 tr3 = (low - close.shift(1)).abs()
                 tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                 atr = tr.ewm(span=14).mean()
            else:
                atr = market_data["atr"]

            future_close = market_data["close"].shift(-lookahead)
            current_close = market_data["close"]

            forward_return_atr = (future_close - current_close) / atr.replace(0, np.nan)

            # Define Classes
            # 0: Down (Return < -0.5 ATR)
            # 1: Neutral
            # 2: Up (Return > 0.5 ATR)
            thresh = float(config.get("volume_force_target_threshold_atr", 0.5))

            y = pd.Series(1, index=market_data.index)  # Default Neutral
            y[forward_return_atr > thresh] = 2  # Up
            y[forward_return_atr < -thresh] = 0 # Down

            # Drop NaN targets
            valid_mask = forward_return_atr.notna() & feature_df.notna().all(axis=1)
            X = feature_df.loc[valid_mask]
            y = y.loc[valid_mask]

            class_counts = y.value_counts().to_dict()
            metrics["class_distribution"] = {str(k): int(v) for k, v in class_counts.items()}
            tprint_info(f"🎯 Target class distribution: {class_counts}")

            # 5. Train Model
            model_id = f"{symbol}_{exchange}_{timeframe}_volume_force"

            training_config = XGBTrainingConfig(
                model_id=model_id,
                retrain_interval_days=10,
                hpo_interval_days=30,
                burnin_pct=1/12,
                n_estimators=500,
                early_stopping_rounds=20,
                tree_method="hist",
                objective="multi:softprob",
                num_class=3,
            )

            trainer = StandardizedXGBTrainer(model_id=model_id, config=training_config)

            tprint_info("🧠 Training XGBoost (Volume Force)...")
            results = trainer.train_and_predict(
                X=X,
                y=y,
                data_start=X.index.min(),
                data_end=X.index.max(),
                eval_metric="mlogloss",
                verbose=True
            )

            # 6. Process Predictions (OOF)
            # Map probabilities to scalar: 0 (Down) -> 0.0, 1 (Neutral) -> 0.5, 2 (Up) -> 1.0
            oof_preds = results.oof_predictions

            prob_cols = [c for c in oof_preds.columns if c.startswith('prob_class_')]
            if len(prob_cols) == 3:
                # Weighted sum: P(Down)*0 + P(Neutral)*0.5 + P(Up)*1.0
                scalar_pred = (
                    oof_preds[prob_cols[0]] * 0.0 +
                    oof_preds[prob_cols[1]] * 0.5 +
                    oof_preds[prob_cols[2]] * 1.0
                )
                oof_preds["scalar_pred"] = scalar_pred

                # Save predictions artifact
                # Ensure the primary output is the scalar prediction aligned to timestamp
                preds_to_save = oof_preds[["scalar_pred"]].reset_index().rename(columns={oof_preds.index.name or "index": "timestamp", "scalar_pred": "predicted"})
                preds_path = self._save_artifact(
                    data=preds_to_save,
                    artifact_name="ml_volume_force_predictions",
                    artifact_type="data",
                    data_category="predictions",
                    metadata={"lookahead": lookahead, "threshold_atr": thresh, "output_type": "scalar_0_1"}
                )
                artifacts.append(preds_path)

                # Also save a dedicated scalar artifact if preferred by downstream conventions,
                # but "ml_volume_force_predictions" with "predicted" column is standard.

                # Metrics
                acc = (oof_preds["pred_class"] == y.loc[oof_preds.index]).mean()
                metrics["accuracy"] = float(acc)
                metrics["scalar_pred_mean"] = float(scalar_pred.mean())
                metrics["scalar_pred_std"] = float(scalar_pred.std())

                # Calculate Log Loss
                # Extract prob columns in order
                prob_cols_sorted = sorted([c for c in oof_preds.columns if c.startswith('prob_class_')])
                y_true = y.loc[oof_preds.index]
                y_probs = oof_preds[prob_cols_sorted].values

                try:
                    ll = log_loss(y_true, y_probs, labels=[0, 1, 2])
                    metrics["oof_log_loss"] = float(ll)
                except Exception as e:
                    tprint_warning(f"Log loss calculation failed: {e}")
                    metrics["oof_log_loss"] = float("inf")

                tprint_success(f"✅ Training complete. OOF Accuracy: {acc:.4f}, LogLoss: {metrics.get('oof_log_loss', 'N/A'):.4f}")
                tprint_info(f"   Scalar prediction stats: Mean={scalar_pred.mean():.3f}, Std={scalar_pred.std():.3f}")
            else:
                tprint_warning("Could not map predictions to scalar (unexpected columns).")

            # Save Model
            if results.models:
                model_path = self._save_artifact(
                    data=results.models[-1],
                    artifact_name="ml_volume_force_model",
                    artifact_type="model",
                    data_category="models",
                    metadata={"n_features": len(X.columns)}
                )
                artifacts.append(model_path)

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
