"""
XGB Meso Trend Step

This step generates "Meso Trend" predictions (typically 2-4 hour horizon) using
an XGBoost model trained on a comprehensive set of technical features (trend,
momentum, volatility, volume, oscillators).

It is fully decoupled from HMM regime discovery steps and generates its own
features directly from OHLCV market data.

Responsibilities:
- Load OHLCV market data.
- Generate technical features using RollingHMMFeatureEngineer (trend, momentum, etc.).
- Generate Cross-Timeframe (HTF) features (RSI, Trend on 1h/4h).
- Compute forward-return-based meso trend targets (volatility-normalized).
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
from src.training.steps.market_analysis.rolling_hmm_clustering.feature_engineering import (
    RollingHMMFeatureEngineer,
    FeatureEngineeringConfig,
    DEFAULT_EWMA_CONFIGS,
    EWMAConfig,
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

            # Defaults for Meso Trend (2-4h horizon)
            meso_defaults: Dict[str, Any] = {
                "meso_trend_target_vol_window": 320,
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
            # 2) Load OHLCV market data
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
            # 3) Feature Engineering (Decoupled from HMM)
            # ------------------------------------------------------------------
            tprint_info("Generating meso trend features (trend, momentum, volatility, volume)...")

            # Configure RollingHMMFeatureEngineer for robust feature generation
            # We use a custom EWMA config targeting the meso timeframe
            # For 15m bars: 12 bars (3h) and 24 bars (6h) seems appropriate for meso
            # For 1h bars: 3 bars and 6 bars

            ewma_short = 12 if 'm' in regime_timeframe else 3
            ewma_long = 24 if 'm' in regime_timeframe else 6

            fe_config = FeatureEngineeringConfig(
                ewma_configs=[EWMAConfig(ewma_short, ewma_long, "meso")],
                use_log_returns=True,
                use_volatility_features=True,
                use_trend_features=True,
                use_volume_features=True,
                normalize_method='robust', # Robust scaling for better stability
                rolling_normalize_window=int(config.get("meso_trend_target_vol_window", 320)),
                enable_vectorbt_optimization=False, # Disable VBT to avoid deps issues
                enable_hardware_optimization=False,
            )

            feature_engineer = RollingHMMFeatureEngineer(fe_config)

            # Generate Base Features
            features_df = feature_engineer.generate_features(
                market_data,
                ewma_config=fe_config.ewma_configs[0],
                use_cache=False
            )

            # Generate Cross-Timeframe Features
            htf_df = self._generate_cross_timeframe_features(market_data, regime_timeframe)

            # Merge Features
            # Join HTF features (they are already reindexed to market_data index)
            if not htf_df.empty:
                aligned_df = features_df.join(htf_df, how="left")
                # Forward fill HTF features to propagate values between HTF bars
                aligned_df.update(aligned_df[htf_df.columns].fillna(method="ffill"))
            else:
                aligned_df = features_df.copy()

            # Append raw Close for target calculation if not present
            aligned_df['close'] = market_data['close']

            # ------------------------------------------------------------------
            # 4) Compute Targets & Final Dataset
            # ------------------------------------------------------------------
            # Compute meso labels (Target: Vol-Normalized Forward Returns)
            # Explicitly configured to target 2-4h horizons (excluding 1h/3h if desired)
            meso_df = self._compute_meso_labels(aligned_df, config)

            if meso_df.empty:
                raise ValueError("Meso dataset is empty after label construction")

            # ------------------------------------------------------------------
            # 5) Train XGBoost Model
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
            # 6) Save Artifacts
            # ------------------------------------------------------------------
            self.set_context(
                symbol=symbol,
                exchange=exchange,
                timeframe=regime_timeframe,
                direction=direction,
                model="regime_alpha", # Keep consistent model namespace for downstream compatibility
            )

            meso_to_save = meso_df.reset_index().rename(
                columns={meso_df.index.name or "index": "timestamp"}
            )

            metadata = {
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": regime_timeframe,
                "source_market_data": market_source,
                "version": "v3_meso_trend_htf",
                "training_start": str(split_config.training.start),
                "training_end": str(split_config.training.effective_end),
                "htf_features_enabled": not htf_df.empty
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

    def _generate_cross_timeframe_features(self, market_data: pd.DataFrame, base_tf: str) -> pd.DataFrame:
        """Generate features from higher timeframes (1h, 4h)."""
        # Only generating for 15m base for now to ensure reliability
        if "15m" not in base_tf:
            tprint_info(f"Skipping HTF features for base timeframe {base_tf} (only 15m supported for now)")
            return pd.DataFrame(index=market_data.index)

        htf_features = pd.DataFrame(index=market_data.index)

        # Define HTFs to process
        htfs = ["1h", "4h"]

        tprint_info(f"Generating HTF features for: {htfs}")

        for htf in htfs:
            try:
                # Resample
                resampled = market_data.resample(htf).agg({
                    "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
                }).dropna()

                if resampled.empty:
                    continue

                # HTF RSI (14 period)
                delta = resampled["close"].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / (loss + 1e-8)
                rsi = 100 - (100 / (1 + rs))

                # HTF Trend (MACD-like: EMA 12 - EMA 26 normalized)
                ema_fast = resampled["close"].ewm(span=12, adjust=False).mean()
                ema_slow = resampled["close"].ewm(span=26, adjust=False).mean()
                trend = (ema_fast - ema_slow) / (resampled["close"] + 1e-8)

                # HTF Volatility (ATR-like normalized by price)
                tr1 = resampled["high"] - resampled["low"]
                tr2 = (resampled["high"] - resampled["close"].shift()).abs()
                tr3 = (resampled["low"] - resampled["close"].shift()).abs()
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr = tr.rolling(14).mean()
                vol = atr / (resampled["close"] + 1e-8)

                # Reindex to base index (broadcast)
                # We reindex to market_data index. Forward fill propagates the HTF value
                # until a new HTF bar closes.

                # Note: To prevent lookahead, we should ideally shift HTF features
                # so that at time T (15m), we only see HTF bars closed BEFORE T.
                # Pandas resample labels by 'left' or 'right'.
                # Standard 'resample' usually labels with the START of the bin or END.
                # If we use closed='right', label='right', the timestamp is the close time.
                # So at 10:00 (15m), we know the 09:00-10:00 1h bar value.
                # Default resample for 1h on 15m data:
                # 09:00, 09:15, 09:30, 09:45 -> 1h bar labelled 09:00 (start) or 10:00 (end).
                # We need to ensure we don't peek.
                # Safest is to rely on ffill. If we have a value at 09:00, it persists.

                aligned_rsi = rsi.reindex(market_data.index, method="ffill")
                aligned_trend = trend.reindex(market_data.index, method="ffill")
                aligned_vol = vol.reindex(market_data.index, method="ffill")

                htf_features[f"htf_{htf}_rsi"] = aligned_rsi
                htf_features[f"htf_{htf}_trend"] = aligned_trend
                htf_features[f"htf_{htf}_vol"] = aligned_vol

            except Exception as e:
                tprint_warning(f"Failed to generate HTF features for {htf}: {e}")
                continue

        return htf_features

    def _compute_meso_labels(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        df = df.copy()
        if "close" not in df.columns:
             raise ValueError("Close price missing for target calculation")

        close = df["close"].astype(float)

        # Calculate Meso Horizons: 2h and 4h (Removed 1h and 3h as requested)
        # Determine multiplier for current timeframe
        multiplier = 1
        tf = config.get("regime_timeframe", "15m")
        if "15m" in tf:
            multiplier = 4
        elif "1h" in tf:
            multiplier = 1
        elif "5m" in tf:
            multiplier = 12

        target_hours = [2, 4]
        fwd_cols = []

        for hours in target_hours:
            bars = hours * multiplier
            fwd_col = f"meso_trend_forward_return_{hours}h"

            # Simple shift target
            df[fwd_col] = np.log(close.shift(-bars) / close)
            fwd_cols.append(fwd_col)

        # Target: Mean of forward returns over selected horizons
        if not fwd_cols:
             raise ValueError("No forward columns generated")

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
        # Exclude targets, forward returns, timestamps, and raw close (we want features only)
        exclude_cols = [c for c in df.columns if "target" in c or "forward_return" in c or "timestamp" in c or c == "close"]

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

        # Apply robust scaling to all features (including new HTF ones)
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
