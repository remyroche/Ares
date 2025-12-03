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
import itertools
import random
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

            # Cap meso_trend_target_vol_window to ~2 days of data (in bars)
            tf_str = str(regime_timeframe)
            try:
                if "m" in tf_str:
                    minutes = int(tf_str.replace("m", ""))
                    bars_per_day = max(int(24 * 60 / max(minutes, 1)), 1)
                elif "h" in tf_str:
                    hours = int(tf_str.replace("h", ""))
                    bars_per_day = max(int(24 / max(hours, 1)), 1)
                else:
                    bars_per_day = 96
            except Exception:
                bars_per_day = 96

            max_vol_bars = 2 * bars_per_day
            raw_vol_window = int(config.get("meso_trend_target_vol_window", 320))
            effective_vol_window = max(1, min(raw_vol_window, max_vol_bars))
            config["meso_trend_target_vol_window"] = effective_vol_window

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

            default_short = 12 if 'm' in regime_timeframe else 3
            default_long = 24 if 'm' in regime_timeframe else 6

            ewma_short = int(config.get("meso_ewma_short", default_short))
            ewma_long = int(config.get("meso_ewma_long", default_long))

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
            htf_df = self._generate_cross_timeframe_features(market_data, regime_timeframe, config)

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

    def generate_config_variations(self, base_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate a grid/random search of feature parameter variations."""

        # Define parameter grid
        variations = {
            # Target Vol Window
            "meso_trend_target_vol_window": [192, 320, 480],

            # EWMA Spans (Short, Long)
            "meso_ewma_short": [8, 12, 16],
            "meso_ewma_long": [16, 24, 32],

            # HTF Lookbacks
            "meso_htf_rsi_period": [14, 21],
            "meso_htf_atr_period": [14, 21],
            "meso_htf_macd_fast": [12, 8],
            "meso_htf_macd_slow": [26, 21],
        }

        # Generate all combinations (Grid Search)
        # Note: We filter for ewma_short < ewma_long

        keys = list(variations.keys())
        values = list(variations.values())

        configs = []
        max_configs = int(base_config.get("meso_sweep_max_configs", 30))

        all_combinations = list(itertools.product(*values))

        # Shuffle for random search effect if limiting count
        random.shuffle(all_combinations)

        for combo in all_combinations:
            if len(configs) >= max_configs:
                break

            cfg_update = dict(zip(keys, combo))

            # Logic constraints
            if cfg_update["meso_ewma_short"] >= cfg_update["meso_ewma_long"]:
                continue

            if cfg_update["meso_htf_macd_fast"] >= cfg_update["meso_htf_macd_slow"]:
                continue

            new_config = base_config.copy()
            new_config.update(cfg_update)
            # Add a signature
            sig_parts = [f"{k}={v}" for k, v in cfg_update.items()]
            new_config["config_signature"] = "|".join(sig_parts)
            configs.append(new_config)

        tprint_info(f"🔧 Generated {len(configs)} configuration variations for sweep")
        return configs

    async def run_config_batch(self, configs: List[Dict[str, Any]], symbol: str, exchange: str) -> List[Dict[str, Any]]:
        """Run a batch of configurations and collect results."""
        results = []
        total = len(configs)

        for i, config in enumerate(configs):
            tprint_info(f"🚀 Running sweep config {i+1}/{total}")

            try:
                # Ensure execution mode is efficient
                config.setdefault("execution_mode", "light")

                # Execute
                result = await self.execute(config)

                metrics = result.get("training_metrics", {})
                success = result.get("success", False)

                # Extract RMSE
                rmse = metrics.get("train_rmse", None) # StandardizedXGBTrainer puts val/oof metric here typically
                target_std = metrics.get("target_std", None)
                zero_baseline_rmse = metrics.get("zero_baseline_rmse", None)
                if rmse is not None and target_std not in (None, 0):
                    rmse_over_target = rmse / target_std
                else:
                    rmse_over_target = None
                if rmse is not None and zero_baseline_rmse is not None:
                    rmse_improvement_vs_zero = zero_baseline_rmse - rmse
                else:
                    rmse_improvement_vs_zero = None

                res_entry = {
                    "config_id": i + 1,
                    "config_signature": config.get("config_signature", "unknown"),
                    "success": success,
                    "rmse": rmse,
                    "target_std": target_std,
                    "zero_baseline_rmse": zero_baseline_rmse,
                    "rmse_over_target_std": rmse_over_target,
                    "rmse_improvement_vs_zero": rmse_improvement_vs_zero,
                    "n_samples": result.get("n_samples", 0),
                    "error": result.get("error", ""),
                }

                # Add config params
                for k in config:
                    if k.startswith("meso_"):
                        res_entry[k] = config[k]

                results.append(res_entry)

                if success:
                    tprint_info(f"✅ Config {i+1} done. RMSE: {rmse}")
                else:
                    tprint_warning(f"⚠️ Config {i+1} failed: {res_entry['error']}")

            except Exception as e:
                tprint_error(f"❌ Config {i+1} crashed: {e}")
                results.append({
                    "config_id": i + 1,
                    "success": False,
                    "error": str(e)
                })

        return results

    def analyze_and_rank_results(self, results: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Rank results by RMSE."""
        df = pd.DataFrame(results)

        if df.empty:
            return df, {}

        successful = df[df["success"] == True].copy()

        if successful.empty:
            return df, {"analysis": "no_successful_runs"}

        # Convert RMSE to float
        successful["rmse"] = pd.to_numeric(successful["rmse"], errors="coerce")
        successful = successful.dropna(subset=["rmse"])

        # Sort ascending (lower RMSE is better)
        successful = successful.sort_values("rmse", ascending=True)

        best_config_row = successful.iloc[0]

        analysis = {
            "best_rmse": float(best_config_row["rmse"]),
            "best_config_id": int(best_config_row["config_id"]),
            "best_signature": best_config_row["config_signature"],
            "total_runs": len(results),
            "successful_runs": len(successful)
        }

        return pd.concat([successful, df[df["success"] == False]], ignore_index=True), analysis

    def _generate_cross_timeframe_features(self, market_data: pd.DataFrame, base_tf: str, config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Generate features from higher timeframes (1h, 4h)."""
        # Only generating for 15m base for now to ensure reliability
        if "15m" not in base_tf:
            tprint_info(f"Skipping HTF features for base timeframe {base_tf} (only 15m supported for now)")
            return pd.DataFrame(index=market_data.index)

        config = config or {}

        htf_features = pd.DataFrame(index=market_data.index)

        # Define HTFs to process
        htfs = ["1h", "4h"]

        # Param lookups
        rsi_period = int(config.get("meso_htf_rsi_period", 14))
        atr_period = int(config.get("meso_htf_atr_period", 14))
        macd_fast = int(config.get("meso_htf_macd_fast", 12))
        macd_slow = int(config.get("meso_htf_macd_slow", 26))

        tprint_info(f"Generating HTF features for: {htfs} with RSI={rsi_period}, ATR={atr_period}, MACD=({macd_fast}, {macd_slow})")

        for htf in htfs:
            try:
                # Resample
                resampled = market_data.resample(htf).agg({
                    "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
                }).dropna()

                if resampled.empty:
                    continue

                # HTF RSI
                delta = resampled["close"].diff()
                gain = (delta.where(delta > 0, 0)).rolling(rsi_period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
                rs = gain / (loss + 1e-8)
                rsi = 100 - (100 / (1 + rs))

                # HTF Trend (MACD-like)
                ema_fast = resampled["close"].ewm(span=macd_fast, adjust=False).mean()
                ema_slow = resampled["close"].ewm(span=macd_slow, adjust=False).mean()
                trend = (ema_fast - ema_slow) / (resampled["close"] + 1e-8)

                # HTF Volatility (ATR-like normalized by price)
                tr1 = resampled["high"] - resampled["low"]
                tr2 = (resampled["high"] - resampled["close"].shift()).abs()
                tr3 = (resampled["low"] - resampled["close"].shift()).abs()
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr = tr.rolling(atr_period).mean()
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
            data_start=X_full_df.index.min(),
            data_end=X_full_df.index.max(),
            eval_metric="rmse",
            verbose=False
        )

        best_model = results.models[-1] if results.models else None

        oof_df = results.oof_predictions if results.oof_predictions is not None else pd.DataFrame()
        if not oof_df.empty:
            pred_col = 'prediction' if 'prediction' in oof_df.columns else (
                'probability' if 'probability' in oof_df.columns else None
            )
            if pred_col is not None:
                scores = oof_df[pred_col]
                # Aggregate any duplicate timestamps by mean to ensure a unique index
                if not scores.index.is_unique:
                    scores = scores.groupby(level=0).mean()
            else:
                scores = pd.Series(dtype=float)
        else:
            scores = pd.Series(dtype=float)

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

        # Derive a simple RMSE metric from deduplicated OOF scores when available
        try:
            if not scores.empty:
                common_idx = scores.index.intersection(y.index)
                if not common_idx.empty:
                    y_oof = y.loc[common_idx].astype(float)
                    y_pred = scores.loc[common_idx].astype(float)
                    rmse = float(np.sqrt(np.mean((y_oof - y_pred) ** 2)))
                else:
                    rmse = float('nan')
            else:
                rmse = float('nan')
        except Exception:
            rmse = float('nan')

        # Baseline diagnostics: target std and zero-prediction RMSE
        if len(y) > 0:
            target_std = float(y.astype(float).std())
            zero_baseline_rmse = float(np.sqrt(np.mean((y.astype(float)) ** 2)))
        else:
            target_std = float('nan')
            zero_baseline_rmse = float('nan')

        metrics = {
            "train_rmse": rmse,
            "target_std": target_std,
            "zero_baseline_rmse": zero_baseline_rmse,
            "n_models": len(results.models),
        }

        feature_pipeline = {
            "feature_names": feature_cols,
            "scaler": scaler,
        }

        return best_model, scores, "meso_trend_score_continuous", metrics, feature_pipeline
