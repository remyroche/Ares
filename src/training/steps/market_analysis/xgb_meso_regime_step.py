"""
XGB Meso Trend Step

This step consumes 1h Rolling HMM regime outputs plus OHLCV data to
construct forward-return-based meso trend labels and trains an XGBoost
model using StandardizedXGBTrainer.

Responsibilities:
- Load 1h HMM artifacts from versioned HDF5 (labels, probabilities,
  economic features).
- Load 1h OHLCV market data.
- Align all series on a common DatetimeIndex.
- Compute forward returns (meso horizon: 1-4h) and binary/continuous targets.
- Train XGBoost model via StandardizedXGBTrainer.
- Save the resulting dataset and model artifacts.
"""

import logging
import time
from typing import Any, Dict, Optional, Tuple, List
from dataclasses import asdict, is_dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from src.training.steps.base_step import BaseStep
from src.utils.tprint import (
    tprint,
    tprint_info,
    tprint_warning,
    tprint_error,
    tprint_success
)
from src.features_common.transforms.scaling_normalization import (
    ScalingNormalizer,
    rolling_adaptive_normalize,
)
from src.utils.versioned_artifacts.temporal_splits import (
    create_temporal_split_config_for_pipeline,
    TemporalSplitConfig,
)
from src.utils.ml_common.standardized_xgb_trainer import (
    StandardizedXGBTrainer,
    XGBTrainingConfig,
)

logger = logging.getLogger(__name__)


class XGBMesoTrendStep(BaseStep):
    """Pipeline step to train XGBoost meso-trend model using StandardizedXGBTrainer."""

    def __init__(self, step_name: str = "xgb_meso_regime"):
        """Initialize the XGB meso trend step with versioned artifacts enabled."""
        super().__init__(step_name, use_versioned_artifacts=True)
        self.logger = logger.getChild("XGBMesoTrendStep") if hasattr(logger, "getChild") else logger
        tprint(f"✅ Initialized {step_name} step", "SUCCESS")

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the meso trend model training."""
        start_time = time.time()

        try:
            symbol = str(config.get("symbol", "ETHUSDT"))
            exchange = str(config.get("exchange", "binance"))
            regime_timeframe = str(
                config.get("regime_timeframe", config.get("timeframe", "15m"))
            )
            direction = str(config.get("direction", "long"))

            # Defaults for Meso Trend (1-4h horizon)
            meso_defaults: Dict[str, Any] = {
                "meso_trend_target_vol_window": 320,
                "meso_trend_enable_trend_features": True,
                "meso_trend_trend_ema_fast_window": 24,
                "meso_trend_trend_ema_slow_window": 64,
                "meso_trend_trend_slope_window": 48,
                "meso_trend_max_horizon_bars": 4,  # 1h bars -> 4h horizon
            }
            for k, v in meso_defaults.items():
                config.setdefault(k, v)

            if not symbol or not exchange:
                raise ValueError("Config must include 'symbol' and 'exchange'")

            tprint_info(
                f"Starting {self.step_name} for {symbol} on {exchange} "
                f"(regime_timeframe={regime_timeframe})"
            )

            # ------------------------------------------------------------------
            # 1) Initialize regime context
            # ------------------------------------------------------------------
            self.set_context(
                symbol=symbol,
                exchange=exchange,
                timeframe=regime_timeframe,
                direction=direction,
                model="regime",
            )

            # ------------------------------------------------------------------
            # 2) Load 1h OHLCV market data
            # ------------------------------------------------------------------
            market_data_config = {
                **config,
                "timeframe": regime_timeframe,
            }
            market_data_load_config = dict(market_data_config)
            market_data_load_config["execution_mode"] = str(
                config.get("execution_mode", "full")
            ).lower()

            market_data, market_source = self.load_market_data_or_fail(
                market_data_load_config,
                pipeline_state={},
                allow_config_override=True,
            )

            if not isinstance(market_data, pd.DataFrame) or market_data.empty:
                raise ValueError("Loaded market data is empty or not a DataFrame")

            if not isinstance(market_data.index, pd.DatetimeIndex):
                try:
                    market_data = market_data.copy()
                    try:
                        market_data.index = pd.to_datetime(market_data.index)
                    except (TypeError, ValueError):
                        market_data.index = pd.to_datetime(market_data.index, utc=True)
                        market_data.index = market_data.index.tz_convert(None)
                except Exception as exc:
                    raise ValueError(
                        "Market data index could not be converted to DatetimeIndex"
                    ) from exc

            tprint_info(
                f"✅ Loaded market data from {market_source}: {market_data.shape} "
                f"({market_data.index.min()} → {market_data.index.max()})"
            )

            # Create temporal split config
            split_config = create_temporal_split_config_for_pipeline(
                symbol=symbol,
                exchange=exchange,
                timeframe=regime_timeframe,
                data_start=market_data.index.min(),
                data_end=market_data.index.max(),
                enable_burnin=True,
            )

            # ------------------------------------------------------------------
            # 3) Obtain HMM artifacts
            # ------------------------------------------------------------------
            # Ensure economic features artifact exists
            try:
                self._get_artifact("rolling_hmm_economic_features", "data")
            except Exception:
                pass

            labels_df, probs_df, economic_df = self._load_hmm_artifacts(config)

            # ------------------------------------------------------------------
            # 4) Align all inputs on common DatetimeIndex
            # ------------------------------------------------------------------
            aligned_df = self._align_inputs(
                market_data=market_data,
                labels_df=labels_df,
                probs_df=probs_df,
                economic_df=economic_df,
            )

            if aligned_df.empty:
                raise ValueError("Aligned dataset is empty after merging inputs")

            # Light-mode filtering
            execution_mode = str(config.get("execution_mode", "full")).lower()
            if execution_mode == "light":
                aligned_df = self._apply_light_mode_filter(
                    aligned_df,
                    config,
                    timeframe=regime_timeframe,
                )
                if aligned_df.empty:
                    raise ValueError("Aligned dataset became empty after light-mode filtering")

            # ------------------------------------------------------------------
            # 5) Feature Engineering & Target Generation
            # ------------------------------------------------------------------
            if bool(config.get("meso_trend_enable_trend_features", True)):
                self._generate_trend_features(aligned_df, config)

            # Compute meso labels
            meso_df = self._compute_meso_labels(aligned_df, config)

            if meso_df.empty:
                raise ValueError("Meso dataset is empty after label construction")

            # ------------------------------------------------------------------
            # 6) Train XGBoost Model
            # ------------------------------------------------------------------
            tprint_info(f"  Starting Meso Trend XGBoost training on {len(meso_df)} samples...")

            model, meso_scores, pred_col_name, training_metrics, feature_pipeline_artifacts = self._train_meso_model(
                meso_df,
                config,
                split_config=split_config,
            )

            if meso_scores is not None:
                meso_df[pred_col_name] = meso_scores
                # Assign to canonical column
                meso_df["meso_trend_score_continuous"] = meso_scores

            # ------------------------------------------------------------------
            # 7) Save Artifacts
            # ------------------------------------------------------------------
            self.set_context(
                symbol=symbol,
                exchange=exchange,
                timeframe=regime_timeframe,
                direction=direction,
                model="regime_alpha",
            )

            meso_to_save = meso_df.reset_index().rename(
                columns={meso_df.index.name or "index": "timestamp"}
            )

            metadata = {
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": regime_timeframe,
                "source_market_data": market_source,
                "version": "v1_meso_trend",
                "training_start": str(split_config.training.start),
                "training_end": str(split_config.training.effective_end),
            }

            training_data_path = self._save_artifact(
                data=meso_to_save,
                artifact_name="xgb_meso_trend_training_data_15m",
                artifact_type="data",
                metadata=metadata,
            )

            model_path = None
            if model is not None:
                model_path = self._save_artifact(
                    data=model,
                    artifact_name="xgb_meso_trend_model_15m",
                    artifact_type="model",
                    metadata={"model_type": "xgboost", **metadata},
                )

            if feature_pipeline_artifacts is not None:
                self._save_artifact(
                    data=feature_pipeline_artifacts,
                    artifact_name="xgb_meso_trend_feature_pipeline_15m",
                    artifact_type="model",
                    metadata=metadata,
                )

            execution_time = time.time() - start_time
            tprint_success(f"✅ {self.step_name} completed in {execution_time:.2f}s")

            return {
                "success": True,
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": regime_timeframe,
                "n_samples": int(len(meso_df)),
                "training_data_path": training_data_path,
                "model_path": model_path,
                "training_metrics": training_metrics,
            }

        except Exception as exc:
            tprint_error(f"❌ {self.step_name} failed: {exc}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(exc)}

    # --------------------------------------------------------------------------
    # Helper Methods
    # --------------------------------------------------------------------------

    def _load_hmm_artifacts(self, config):
        labels = self._get_artifact("rolling_hmm_regime_labels", "data")
        probs = self._get_artifact("rolling_hmm_regime_probabilities", "data")
        economic = self._get_artifact("rolling_hmm_economic_features", "data")

        def _prep(a):
            if a is None: return None
            df = pd.DataFrame(a)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.set_index("timestamp")
            return df

        return _prep(labels), _prep(probs), _prep(economic)

    def _align_inputs(self, market_data, labels_df, probs_df, economic_df):
        aligned = market_data.copy()
        if "close" not in aligned.columns:
            aligned.columns = [c.lower() for c in aligned.columns]

        for df in [labels_df, probs_df, economic_df]:
            if df is not None and not df.empty:
                aligned = aligned.join(df, how="inner")
        return aligned.dropna(how="all")

    def _generate_trend_features(self, df: pd.DataFrame, config: Dict[str, Any]):
        if "close" not in df.columns:
            return
        close = df["close"].astype(float)

        fast_w = int(config.get("meso_trend_trend_ema_fast_window", 24))
        slow_w = int(config.get("meso_trend_trend_ema_slow_window", 64))
        slope_w = int(config.get("meso_trend_trend_slope_window", 48))

        if fast_w > 0:
            df["meso_trend_ema_fast"] = close.ewm(span=fast_w, adjust=False).mean()
        if slow_w > 0:
            df["meso_trend_ema_slow"] = close.ewm(span=slow_w, adjust=False).mean()

        if slope_w > 1:
            log_price = np.log(close.clip(lower=1e-8))
            shifts = log_price.shift(slope_w)
            df["meso_trend_price_slope"] = (log_price - shifts) / slope_w

    def _compute_meso_labels(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        df = df.copy()
        close = df["close"].astype(float)

        max_h = int(config.get("meso_trend_max_horizon_bars", 4))
        fwd_cols = []
        for h in range(1, max_h + 1):
            fwd_col = f"meso_trend_forward_return_{h}h"

            # Determine shift based on timeframe. Assuming market data index reflects timeframe.
            # If 15m data -> shift(-4) is 1h. If 1h data -> shift(-1) is 1h.
            # hmm_macro_regime assumes data is at 'regime_timeframe' (1h).
            # So we use shift(-h).

            regime_tf = config.get("regime_timeframe", "1h")
            multiplier = 1
            if "15m" in regime_tf:
                multiplier = 4
            elif "5m" in regime_tf:
                multiplier = 12

            shift_val = -1 * h * multiplier
            df[fwd_col] = np.log(close.shift(shift_val) / close)
            fwd_cols.append(fwd_col)

        # Target: Mean of forward returns over horizon
        df["meso_trend_target_raw"] = df[fwd_cols].mean(axis=1)

        # Volatility normalization
        vol_window = int(config.get("meso_trend_target_vol_window", 320))
        returns = np.log(close).diff()
        vol = returns.rolling(vol_window).std()
        df["meso_trend_target_vol"] = vol

        df["meso_trend_target"] = df["meso_trend_target_raw"] / (vol + 1e-8)

        return df.dropna(subset=["meso_trend_target"])

    def _train_meso_model(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any],
        split_config: Optional[TemporalSplitConfig],
    ):
        target_col = "meso_trend_target"
        exclude_cols = [c for c in df.columns if "target" in c or "forward_return" in c or "timestamp" in c]
        # Exclude other metadata cols
        feature_cols = [c for c in df.select_dtypes(include=np.number).columns if c not in exclude_cols]

        X = df[feature_cols]
        y = df[target_col]

        if split_config:
            train_mask = (X.index >= split_config.training.start) & (X.index <= split_config.training.effective_end)
            val_mask = (X.index >= split_config.validation.start) & (X.index <= split_config.validation.effective_end)
            X_train = X[train_mask]
            y_train = y[train_mask]
            X_val = X[val_mask]
            y_val = y[val_mask]
        else:
            split_idx = int(len(X) * 0.8)
            X_train = X.iloc[:split_idx]
            y_train = y.iloc[:split_idx]
            X_val = X.iloc[split_idx:]
            y_val = y.iloc[split_idx:]

        scaler = ScalingNormalizer({"default_strategy": "robust"})
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_full_scaled = scaler.transform(X)

        model_id = f"{config.get('symbol')}_meso_trend"
        trainer_config = XGBTrainingConfig(
            model_id=model_id,
            task_type="regression",
            objective="reg:squarederror",
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            early_stopping_rounds=30,
        )

        trainer = StandardizedXGBTrainer(model_id, trainer_config)

        X_full_df = pd.DataFrame(X_full_scaled, index=X.index, columns=feature_cols)

        results = trainer.train_and_predict(
            X=X_full_df,
            y=y,
            eval_metric="rmse",
            verbose=False
        )

        best_model = results.models[-1] if results.models else None
        scores = results.oof_predictions['prediction'] if results.oof_predictions is not None else pd.Series(dtype=float)

        # Fill missing scores with model prediction (retrodiction)
        if scores.empty and best_model:
             import xgboost as xgb
             dtest = xgb.DMatrix(X_full_scaled, feature_names=feature_cols)
             preds = best_model.predict(dtest)
             scores = pd.Series(preds, index=X.index)
        elif best_model:
             # Align indices
             scores = scores.reindex(X.index)
             mask_nan = scores.isna()
             if mask_nan.any():
                 import xgboost as xgb
                 dtest = xgb.DMatrix(X_full_scaled[mask_nan], feature_names=feature_cols)
                 preds = best_model.predict(dtest)
                 scores[mask_nan] = preds

        metrics = {
            "train_rmse": results.metrics.get("rmse_mean"),
            "n_models": len(results.models),
        }

        feature_pipeline = {
            "feature_names": feature_cols,
            "scaler": scaler
        }

        return best_model, scores, "meso_trend_score_continuous", metrics, feature_pipeline
