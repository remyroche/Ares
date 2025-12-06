"""
ML SMC Regime Step

Smart Money Concepts (SMC) step that consumes 15m OHLCV data to construct
SMC-based features (liquidity scalars, FVG/inefficiency, premium/discount,
displacement, volume profile, time categories) and trains an XGBoost classifier
to predict ATR-normalized forward returns with conformal prediction calibration.
"""
import logging
import time
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List, Union

import numpy as np
import pandas as pd

from src.training.steps.base_step import BaseStep
from src.utils.tprint import (
    tprint,
    tprint_info,
    tprint_warning,
    tprint_error,
    tprint_success,
)
from src.features_common.transforms.scaling_normalization import (
    winsorized_zscore_normalize,
    rolling_adaptive_normalize,
)
from src.utils.feature_common.volume_transforms import log1p_zscore_normalize
from src.feature_generation.categories.smc_regime_features import (
    generate_smc_regime_features,
)
from src.utils.ml_common.evaluation.hsic import calculate_hsic
from sklearn.isotonic import IsotonicRegression
from src.utils.versioned_artifacts.temporal_splits import (
    create_temporal_split_config_for_pipeline,
    TemporalSplitConfig,
)
from src.utils.ml_common.standardized_xgb_trainer import (
    StandardizedXGBTrainer,
    XGBTrainingConfig,
    XGBTrainingResults,
)

logger = logging.getLogger(__name__)


class MLSMCRegimeStep(BaseStep):
    def __init__(self, step_name: str = "ml_smc_regime_step"):
        super().__init__(step_name, use_versioned_artifacts=True)
        self.logger = logger.getChild("MLSMCRegimeStep") if hasattr(logger, "getChild") else logger
        self._cached_market_data_15m = None
        self._cached_market_source_15m = None
        self._cached_market_cache_key_15m = None
        tprint(f"✅ Initialized {step_name} step", "SUCCESS")

    def get_config_signature(self, config: Dict[str, Any]) -> str:
        """Generate a compact signature for configuration identification."""
        key_params = [
            "smc_lookahead",
            "smc_breakout_threshold",
            "smc_breakdown_threshold",
            "smc_vp_lookback",
            "smc_xgb_max_features",
            "smc_normalization_window"
        ]

        parts = []
        for param in key_params:
            value = config.get(param, "default")
            # Format floats compactly
            if isinstance(value, float):
                val_str = f"{value:.5f}".rstrip("0").rstrip(".")
            else:
                val_str = str(value)
            parts.append(f"{param.replace('smc_', '')}={val_str}")

        return "|".join(parts)

    async def run_config_batch(self, configs: List[Dict[str, Any]], symbol: str, exchange: str) -> List[Dict[str, Any]]:
        """Run a batch of configurations and collect results."""

        results = []
        total_configs = len(configs)

        for i, base_config in enumerate(configs):
            # Ensure critical keys are set
            config = dict(base_config)
            config["symbol"] = symbol
            config["exchange"] = exchange

            tprint_info(f"🚀 Running config {i+1}/{total_configs}: {self.get_config_signature(config)}")

            try:
                # Run the step with this configuration
                start_time = time.time()
                result = await self.execute(config)
                execution_time = time.time() - start_time

                # Extract key metrics
                metrics = result.get("metrics", {})
                quality_metrics = {
                    "config_signature": self.get_config_signature(config),
                    "config_id": i + 1,
                    "execution_time": execution_time,
                    "success": result.get("success", False),
                    # Primary Objectives (IC + downstream trading metrics)
                    "smc_xgb_oof_ic": metrics.get("smc_xgb_oof_ic", float("nan")),
                    "smc_xgb_oof_sharpe_gated_25pct": metrics.get("smc_xgb_oof_sharpe_gated_25pct", float("-inf")),
                    "smc_xgb_oof_logloss": metrics.get("smc_xgb_oof_logloss", float("inf")),
                    "smc_xgb_oof_accuracy": metrics.get("smc_xgb_oof_accuracy", 0.0),
                    "error": result.get("error", ""),
                }

                # Add configuration details
                quality_metrics.update({
                    f"config_{k}": v for k, v in config.items()
                    if k.startswith("smc_") and not callable(v)
                })

                results.append(quality_metrics)

                if result.get("success", False):
                    ic = quality_metrics['smc_xgb_oof_ic']
                    tprint_info(f"✅ Config {i+1} completed: OOF IC={ic:.4f}")
                else:
                    tprint_warning(f"⚠️ Config {i+1} failed: {quality_metrics['error']}")

            except Exception as e:
                tprint_error(f"❌ Config {i+1} crashed: {e}")
                results.append({
                    "config_signature": self.get_config_signature(config),
                    "config_id": i + 1,
                    "execution_time": 0,
                    "success": False,
                    "error": str(e),
                    "smc_xgb_oof_ic": float("nan"),
                })

        return results

    def analyze_and_rank_results(self, results: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Analyze results and rank configurations by quality (OOF IC)."""

        if not results:
            return pd.DataFrame(), {}

        df = pd.DataFrame(results)

        # Filter successful runs
        successful = df[df["success"] == True].copy()
        failed = df[df["success"] == False].copy()

        tprint_info(f"📊 Analysis: {len(successful)} successful, {len(failed)} failed runs")

        if len(successful) == 0:
            tprint_warning("⚠️ No successful configurations to analyze")
            return df, {"best_config": None, "analysis": "no_successful_runs"}

        # Rank by IC (Information Coefficient)
        successful = successful.sort_values("smc_xgb_oof_ic", ascending=False)

        # Get best configuration
        best_config = successful.iloc[0].to_dict()

        # Analysis summary
        analysis = {
            "best_config": best_config,
            "total_runs": len(results),
            "successful_runs": len(successful),
            "failed_runs": len(failed),
            "best_ic": float(best_config.get("smc_xgb_oof_ic", np.nan)),
            "best_sharpe_gated": float(best_config.get("smc_xgb_oof_sharpe_gated_25pct", np.nan)),
            "best_accuracy": float(best_config.get("smc_xgb_oof_accuracy", np.nan)),
            "ranking_metric": "smc_xgb_oof_ic",
            "top_5_configs": successful.head(5).to_dict("records"),
            "parameter_importance": self.analyze_parameter_importance(successful),
        }

        # Display results
        self.display_results_summary(successful, failed, analysis)

        return pd.concat([successful, failed], ignore_index=True), analysis

    def analyze_parameter_importance(self, successful: pd.DataFrame) -> Dict[str, Any]:
        """Analyze which parameters correlate with better results."""

        importance = {}

        # Analyze key parameters
        key_params = [
            "config_smc_lookahead",
            "config_smc_breakout_threshold",
            "config_smc_breakdown_threshold",
            "config_smc_vp_lookback",
            "config_smc_xgb_max_features",
            "config_smc_normalization_window"
        ]

        for param in key_params:
            if param in successful.columns:
                # Group by parameter value and compute mean scores
                param_analysis = successful.groupby(param)["smc_xgb_oof_ic"].agg([
                    "count", "mean", "std", "min", "max"
                ]).round(4)
                importance[param] = param_analysis.to_dict()

        return importance

    def display_results_summary(self, successful: pd.DataFrame, failed: pd.DataFrame, analysis: Dict[str, Any]) -> None:
        """Display comprehensive results summary."""

        print("\n" + "="*80)
        print("🏆 SMC REGIME CONFIGURATION OPTIMIZATION RESULTS")
        print("="*80)

        print(f"\n📊 SUMMARY:")
        print(f"   Total configurations tested: {analysis['total_runs']}")
        print(f"   Successful runs: {analysis['successful_runs']}")
        print(f"   Failed runs: {analysis['failed_runs']}")
        print(f"   Success rate: {analysis['successful_runs']/analysis['total_runs']*100:.1f}%")

        if analysis['best_config']:
            print(f"\n🥇 BEST CONFIGURATION:")
            print(f"   Signature: {analysis['best_config']['config_signature']}")
            print(f"   OOF IC: {analysis['best_ic']:.4f}")
            print(f"   OOF Sharpe (Gated): {analysis['best_sharpe_gated']:.4f}")
            print(f"   Execution Time: {analysis['best_config']['execution_time']:.1f}s")

        print(f"\n🏅 TOP 5 CONFIGURATIONS:")
        cols = [
            "config_id", "config_signature", "smc_xgb_oof_ic",
            "smc_xgb_oof_sharpe_gated_25pct", "execution_time"
        ]
        # Ensure cols exist
        avail_cols = [c for c in cols if c in successful.columns]
        print(successful[avail_cols].head(5).to_string(index=False))

        # Parameter importance analysis
        if analysis["parameter_importance"]:
            print(f"\n🔍 PARAMETER IMPORTANCE (by OOF IC):")
            for param, stats in analysis["parameter_importance"].items():
                param_name = param.replace("config_", "")
                print(f"\n   {param_name}:")
                for value, metrics in stats.items():
                    if isinstance(metrics, dict) and "count" in metrics:
                        print(f"      {value}: ic={metrics['mean']:.3f} (count={metrics['count']})")

        if len(failed) > 0:
            print(f"\n❌ COMMON FAILURE MODES:")
            if "error" in failed.columns:
                error_counts = failed["error"].value_counts().head(5)
                for error, count in error_counts.items():
                    print(f"   {count}x: {str(error)[:100]}...")

        print("\n" + "="*80)

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        artifacts: List[str] = []
        metrics: Dict[str, Any] = {}

        try:
            symbol = str(config.get("symbol", "ETHUSDT"))
            exchange = str(config.get("exchange", "binance"))
            regime_timeframe = str(config.get("regime_timeframe", config.get("timeframe", "15m")))
            direction = str(config.get("direction", "long"))

            if not symbol or not exchange:
                raise ValueError("Config must include 'symbol' and 'exchange'")

            # Route all SMC artifacts into a dedicated SMC regime namespace
            self.set_context(
                symbol=symbol,
                exchange=exchange,
                timeframe=regime_timeframe,
                direction=direction,
                model="smc_regime",
            )

            tprint_info(
                f"🚀 Starting {self.step_name} (SMC Regime) for {symbol} on {exchange} "
                f"(regime_timeframe={regime_timeframe})"
            )

            # Load market data with caching (using BaseStep pattern)
            exec_mode_cfg = str(config.get("execution_mode", "")).lower()
            cache_key = (symbol, exchange, regime_timeframe, exec_mode_cfg, "smc")
            market_data = None
            market_source = None

            if (
                getattr(self, "_cached_market_data_15m", None) is not None
                and getattr(self, "_cached_market_cache_key_15m", None) == cache_key
            ):
                try:
                    market_data = self._cached_market_data_15m.copy()
                except Exception:
                    market_data = self._cached_market_data_15m
                market_source = self._cached_market_source_15m
                tprint_info("♻️ Reusing cached 15m market data for ML SMC regimes")
            else:
                market_data, market_source = self.load_market_data_or_fail(
                    {**config, "timeframe": regime_timeframe},
                    pipeline_state={},
                    allow_config_override=True,
                    light_mode_filter=False,
                    skip_artifacts=True,
                )
                if isinstance(market_data, pd.DataFrame):
                    self._cached_market_data_15m = market_data.copy()
                else:
                    self._cached_market_data_15m = market_data
                self._cached_market_source_15m = market_source
                self._cached_market_cache_key_15m = cache_key

            if not isinstance(market_data, pd.DataFrame) or market_data.empty:
                raise ValueError("Loaded market data is empty or not a DataFrame")

            if not isinstance(market_data.index, pd.DatetimeIndex):
                market_data = market_data.copy()
                market_data.index = pd.to_datetime(market_data.index)
                if market_data.index.tz is not None:
                    market_data.index = market_data.index.tz_convert(None)

            market_data = market_data.sort_index()

            tprint_info(
                f"✅ Loaded market data from {market_source}: {market_data.shape} "
                f"({market_data.index.min()} → {market_data.index.max()})"
            )

            # Create temporal split config with 6-month burn-in for indicator stabilization
            split_config = create_temporal_split_config_for_pipeline(
                symbol=symbol,
                exchange=exchange,
                timeframe=regime_timeframe,
                data_start=market_data.index.min(),
                data_end=market_data.index.max(),
                enable_burnin=True,
                # Use default burnin_pct=1/12 (3 months)
            )
            tprint_info(
                f"📊 Temporal split config created with burn-in: "
                f"Burn-in {split_config.burnin.start if split_config.burnin else 'N/A'} → "
                f"{split_config.burnin.effective_end if split_config.burnin else 'N/A'}, "
                f"Train {split_config.training.start} → {split_config.training.effective_end}, "
                f"Val {split_config.validation.start} → {split_config.validation.effective_end}, "
                f"Test {split_config.test.start} → {split_config.test.effective_end}"
            )

            # Generate SMC features with proper normalization
            smc_df = self._generate_smc_features(market_data, config)

            # Train XGBoost model if enabled
            if bool(config.get("smc_xgb_enable_training", True)):
                try:
                    xgb_metrics, xgb_artifacts = self._train_smc_xgb_oof(
                        smc_df,
                        config,
                        split_config,
                        symbol,
                        exchange,
                        regime_timeframe,
                    )
                    if xgb_metrics:
                        metrics.update(xgb_metrics)
                    if xgb_artifacts:
                        artifacts.extend(xgb_artifacts)
                except Exception as xgb_exc:
                    tprint_error(f"SMC XGB OOF training failed: {xgb_exc}")
                    raise

            # Save SMC features using BaseStep pattern
            idx_name = smc_df.index.name or "index"
            if idx_name in smc_df.columns:
                to_save = smc_df.copy()
                if "timestamp" not in to_save.columns:
                    to_save.insert(0, "timestamp", smc_df.index.to_numpy())
            else:
                to_save = smc_df.reset_index().rename(columns={idx_name: "timestamp"})

            smc_features_path = self._save_artifact(
                data=to_save,
                artifact_name="ml_smc_features_15m",
                artifact_type="data",
                data_category="features",
                metadata={
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": regime_timeframe,
                    "source_market_data": market_source,
                },
            )
            artifacts.append(smc_features_path)

            metrics.update({
                "n_samples": int(len(smc_df)),
            })

            execution_time = time.time() - start_time
            metrics["execution_time"] = execution_time

            tprint_success(
                f"✅ {self.step_name} (SMC Regime) completed in {execution_time:.2f}s "
                f"with {len(smc_df)} samples"
            )

            return {
                "success": True,
                "artifacts": artifacts,
                "metrics": metrics,
                "error": None,
                "execution_time": execution_time,
            }

        except Exception as exc:
            execution_time = time.time() - start_time
            tprint_error(f"❌ {self.step_name} failed: {exc}")
            return {
                "success": False,
                "artifacts": artifacts,
                "metrics": metrics,
                "error": str(exc),
                "execution_time": execution_time,
            }

    def _generate_smc_features(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any],
    ) -> pd.DataFrame:
        """Generate all SMC features with proper normalization.

        Delegates core SMC feature construction to the feature bank
        (``generate_smc_regime_features``) so that feature definitions
        are centralized. This method then applies step-specific
        normalization.
        """

        result = generate_smc_regime_features(df, config)

        # Apply normalization to features (except categorical/binary features)
        tprint_info("Normalizing SMC features...")
        result = self._normalize_smc_features(result, config)

        tprint_success(f"✅ Generated {len(result.columns)} SMC features")

        return result

    def _normalize_smc_features(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Apply winsorized z-score normalization to continuous features, log1p to volume features."""

        # Identify continuous features (exclude binary/categorical and raw price levels)
        exclude_patterns = [
            "_session_", "_dow_", "_market_structure_", "_is_", "_confirmed",
            "smc_pdh", "smc_pdl", "smc_day_open", "smc_week_open", "atr",
            "open", "high", "low", "close", "volume"
        ]

        continuous_features = []
        volume_features = []

        for col in df.columns:
            if any(pattern in col for pattern in exclude_patterns):
                continue
            if col.startswith("smc_"):
                if "volume" in col.lower() or "rel_volume" in col.lower():
                    volume_features.append(col)
                else:
                    continuous_features.append(col)

        # Apply rolling z-score normalization to continuous features to prevent look-ahead bias
        if continuous_features:
            window_size = int(config.get("smc_normalization_window", 500))
            tprint_info(
                f"Applying rolling winsorized z-score normalization to {len(continuous_features)} features (window={window_size})"
            )
            try:
                df[continuous_features] = winsorized_zscore_normalize(
                    df[continuous_features],
                    window=window_size,
                )
            except Exception as e:
                tprint_warning(f"Failed to normalize continuous SMC features: {e}")

        # Apply log1p + rolling z-score normalization to volume features
        if volume_features:
            window_size = int(config.get("smc_normalization_window", 500))
            tprint_info(f"Applying log1p+zscore normalization to {len(volume_features)} volume features")
            try:
                df[volume_features] = rolling_adaptive_normalize(
                    df[volume_features],
                    window=window_size,
                    min_periods=window_size // 2,
                    high=df["high"] if "high" in df.columns else None,
                    low=df["low"] if "low" in df.columns else None,
                    close=df["close"] if "close" in df.columns else None,
                    volume_columns=volume_features,
                )
            except Exception as e:
                tprint_warning(
                    f"Failed to log1p+zscore normalize SMC volume features via adaptive normalizer: {e}"
                )
                for feat in volume_features:
                    try:
                        df[feat] = log1p_zscore_normalize(
                            df[feat].clip(lower=0.0),
                            window=window_size,
                            min_periods=window_size // 2,
                        )
                    except Exception as inner_e:
                        tprint_warning(f"Failed to log1p+zscore normalize {feat}: {inner_e}")

        return df

    def _train_smc_xgb_oof(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any],
        split_config: TemporalSplitConfig,
        symbol: str,
        exchange: str,
        regime_timeframe: str,
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Train XGBoost classifier with OOF predictions using standardized trainer.

        This replaces the old _train_smc_xgb_model with proper OOF predictions.
        No data leakage - only returns predictions on data the model hasn't seen.
        """
        metrics: Dict[str, Any] = {}
        artifacts: List[str] = []

        try:
            import xgboost as xgb
        except Exception:
            tprint_warning("xgboost not installed, skipping SMC XGB training")
            metrics["smc_xgb_early_exit_reason"] = "xgboost_not_installed"
            return metrics, artifacts

        if "close" not in df.columns or "high" not in df.columns or "low" not in df.columns:
            tprint_warning("SMC XGB: missing price columns; skipping training")
            metrics["smc_xgb_early_exit_reason"] = "missing_price_columns"
            return metrics, artifacts

        # Create ATR-normalized forward return target
        lookahead = int(config.get("smc_lookahead", 16))

        future_close = df["close"].shift(-lookahead)
        current_close = df["close"]
        atr = df["atr"].replace(0, 0.0001)

        target_atr_return = (future_close - current_close) / atr
        atr_clip = float(config.get("smc_target_atr_clip", 3.0))
        if atr_clip > 0.0:
            target_atr_return = target_atr_return.clip(-atr_clip, atr_clip)

        df_with_target = df.copy()
        df_with_target["target_atr_return"] = target_atr_return
        df_with_target["forward_return"] = (future_close / current_close - 1.0)
        df_with_target = df_with_target.dropna(subset=["target_atr_return", "forward_return"])

        tprint_info(
            f"SMC XGB: target ATR-normalized return stats - mean: {df_with_target['target_atr_return'].mean():.3f}, "
            f"std: {df_with_target['target_atr_return'].std():.3f}, "
            f"range: [{df_with_target['target_atr_return'].min():.3f}, {df_with_target['target_atr_return'].max():.3f}]"
        )

        # Derive 3-class labels: 0=breakdown, 1=neutral, 2=breakout
        breakout_threshold = float(config.get("smc_breakout_threshold", 0.5))
        breakdown_threshold = float(config.get("smc_breakdown_threshold", -0.5))

        target_clean = df_with_target["target_atr_return"]
        target_class = np.where(
            target_clean > breakout_threshold,
            2,
            np.where(target_clean < breakdown_threshold, 0, 1),
        ).astype(np.int32)
        df_with_target["target_class"] = target_class

        class_counts = {
            "breakdown": int((target_class == 0).sum()),
            "neutral": int((target_class == 1).sum()),
            "breakout": int((target_class == 2).sum()),
        }
        tprint_info(
            "SMC XGB: class distribution (0=breakdown,1=neutral,2=breakout): "
            f"{class_counts}"
        )

        # Select features
        exclude_cols = [
            "target_atr_return",
            "forward_return",
            "target_class",
            "smc_pdh",
            "smc_pdl",
            "smc_day_open",
            "smc_week_open",
        ]
        numeric_df = df_with_target.select_dtypes(include=[np.number])
        feature_cols = [col for col in numeric_df.columns if col not in exclude_cols and col.startswith("smc_")]

        if len(feature_cols) < 5:
            tprint_warning(f"SMC XGB: insufficient features (n={len(feature_cols)})")
            metrics["smc_xgb_early_exit_reason"] = f"insufficient_features_{len(feature_cols)}"
            return metrics, artifacts

        max_features = int(config.get("smc_xgb_max_features", 48))
        if len(feature_cols) > max_features:
            target_for_corr = df_with_target["target_atr_return"]
            corr_scores: List[Tuple[str, float]] = []
            for col in feature_cols:
                try:
                    corr_val = numeric_df[col].corr(target_for_corr)
                except Exception:
                    corr_val = 0.0
                if corr_val is None or not np.isfinite(corr_val):
                    corr_val = 0.0
                corr_scores.append((col, float(abs(corr_val))))
            corr_scores.sort(key=lambda x: x[1], reverse=True)
            selected = [name for name, _ in corr_scores[:max_features]]
            tprint_info(
                "SMC XGB: reducing feature set from "
                f"{len(feature_cols)} to {len(selected)} based on correlation with target_atr_return"
            )
            feature_cols = selected

        X = numeric_df[feature_cols].astype(np.float32)
        y_class = df_with_target["target_class"].astype(np.int32)

        # Handle infinities and NaNs
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0.0)

        min_samples = int(config.get("smc_xgb_min_samples", 800))
        if len(X) < min_samples:
            tprint_warning(f"SMC XGB: insufficient samples (n={len(X)}, min={min_samples})")
            metrics["smc_xgb_early_exit_reason"] = f"insufficient_samples_{len(X)}"
            return metrics, artifacts

        tprint_info(
            f"📊 Training SMC XGB with OOF predictions: {len(X)} samples, {len(feature_cols)} features"
        )

        # Create model ID
        model_id = f"{symbol}_{exchange}_{regime_timeframe}_smc"

        # Create training config with multi-class classification
        training_config = XGBTrainingConfig(
            model_id=model_id,
            retrain_interval_days=10,
            hpo_interval_days=30,
            burnin_pct=1/12,
            n_estimators=500,
            hpo_n_estimators=300,
            early_stopping_rounds=20,
            # Multi-class classification (3 classes: breakdown, neutral, breakout)
            tree_method="hist",
            objective="multi:softprob",
            num_class=3,
        )

        # Create trainer
        trainer = StandardizedXGBTrainer(model_id=model_id, config=training_config)

        tprint_info("Training XGBoost with StandardizedXGBTrainer (multi-class: breakdown/neutral/breakout)...")
        results = trainer.train_and_predict(
            X=X,
            y=y_class,
            data_start=df_with_target.index.min(),
            data_end=df_with_target.index.max(),
            eval_metric="mlogloss",
            verbose=True
        )

        # Extract OOF predictions
        oof_predictions = results.oof_predictions
        oof_models = results.models
        oof_metadata = results.metadata

        # Convert multi-class probabilities to scalar (0=downtrend, 1=uptrend)
        # For 3 classes: [breakdown, neutral, breakout] -> scalar using [0.0, 0.5, 1.0]
        tprint_info(f"Converting {len(oof_predictions)} multi-class OOF predictions to scalar...")

        # Extract class probabilities
        prob_cols = [c for c in oof_predictions.columns if c.startswith('prob_class_')]
        if len(prob_cols) != 3:
            raise ValueError(f"Expected 3 class probabilities, got {len(prob_cols)}")

        # Create class-to-scalar mapping: class 0=0.0, class 1=0.5, class 2=1.0
        class_to_scalar = np.array([0.0, 0.5, 1.0], dtype=np.float32)

        # Convert probabilities to scalar: scalar = sum(prob_i * scalar_i)
        proba_matrix = oof_predictions[prob_cols].values
        scalar_predictions = proba_matrix.dot(class_to_scalar)

        # Add scalar to predictions dataframe
        oof_predictions['scalar'] = scalar_predictions

        # Calculate metrics on OOF predictions
        tprint_info(f"Calculating metrics on {len(oof_predictions)} OOF predictions...")

        # Align predictions with targets
        aligned_df = pd.DataFrame(index=df_with_target.index)
        aligned_df["y_true_class"] = y_class
        aligned_df["y_true_atr_return"] = df_with_target["target_atr_return"]
        aligned_df["forward_return"] = df_with_target["forward_return"]
        aligned_df = aligned_df.join(oof_predictions, how='left')

        # Only calculate metrics on OOF samples (where we have predictions)
        oof_mask = ~aligned_df["scalar"].isna()
        n_oof = oof_mask.sum()

        if n_oof > 0:
            y_true_class_oof = aligned_df.loc[oof_mask, "y_true_class"]
            y_true_atr_oof = aligned_df.loc[oof_mask, "y_true_atr_return"]
            y_pred_scalar_oof = aligned_df.loc[oof_mask, "scalar"]
            proba_oof = aligned_df.loc[oof_mask, prob_cols].values
            fwd_ret_oof = aligned_df.loc[oof_mask, "forward_return"]

            from sklearn.metrics import (
                accuracy_score,
                f1_score,
                log_loss,
                mean_squared_error,
                r2_score,
                brier_score_loss,
            )

            # Class predictions
            y_pred_cls_oof = np.argmax(proba_oof, axis=1)

            # Classification metrics
            oof_accuracy = accuracy_score(y_true_class_oof, y_pred_cls_oof)
            oof_f1 = f1_score(y_true_class_oof, y_pred_cls_oof, average="macro")
            oof_logloss = log_loss(y_true_class_oof, proba_oof, labels=[0, 1, 2])

            # Regression-style metrics (scalar vs ATR-normalized return)
            oof_rmse = np.sqrt(mean_squared_error(y_true_atr_oof, y_pred_scalar_oof))
            try:
                oof_r2 = r2_score(y_true_atr_oof, y_pred_scalar_oof)
            except Exception:
                oof_r2 = float("nan")

            # Information coefficient
            try:
                oof_ic = float(np.corrcoef(y_pred_scalar_oof, y_true_atr_oof)[0, 1])
            except Exception:
                oof_ic = float("nan")

            # Gated Sharpe (Top 25% confidence) - Informational purpose
            try:
                # Confidence = abs(scalar - 0.5) * 2 (0.0 to 1.0)
                confidence = np.abs(y_pred_scalar_oof - 0.5) * 2.0

                # Use robust quantile to find threshold for top 25%
                threshold = np.nanquantile(confidence, 0.75)
                gated_mask = confidence >= threshold

                if gated_mask.sum() >= 20:
                    # Directional signal: if scalar > 0.5, Long; if < 0.5, Short
                    direction = np.sign(y_pred_scalar_oof - 0.5)
                    # For perfectly neutral (0.5), direction is 0

                    gated_returns = fwd_ret_oof[gated_mask] * direction[gated_mask]
                    mean_ret = np.mean(gated_returns)
                    std_ret = np.std(gated_returns, ddof=1)

                    if std_ret > 0:
                        oof_sharpe_gated = mean_ret / std_ret
                    else:
                        oof_sharpe_gated = float("-inf")
                else:
                    oof_sharpe_gated = float("-inf")
            except Exception as e:
                tprint_warning(f"Failed to calculate gated Sharpe: {e}")
                oof_sharpe_gated = float("-inf")

            metrics.update({
                "smc_xgb_oof_accuracy": float(oof_accuracy),
                "smc_xgb_oof_f1": float(oof_f1),
                "smc_xgb_oof_logloss": float(oof_logloss),
                "smc_xgb_oof_rmse": float(oof_rmse),
                "smc_xgb_oof_r2": float(oof_r2) if np.isfinite(oof_r2) else float("nan"),
                "smc_xgb_oof_ic": float(oof_ic) if np.isfinite(oof_ic) else float("nan"),
                "smc_xgb_oof_sharpe_gated_25pct": float(oof_sharpe_gated),
                "smc_xgb_oof_samples": int(n_oof),
                "smc_xgb_oof_windows": len(oof_metadata),
                "smc_xgb_hpo_runs": sum(1 for m in oof_metadata if m.get('used_hpo', False)),
            })

        # Derive full-sample predictions and classical train/val/test metrics for reporting
        report_y_pred_scalar = None
        report_y_pred_class = None
        report_feature_importances = None
        train_rmse = val_rmse = test_rmse = float("nan")
        train_r2 = val_r2 = test_r2 = float("nan")
        train_brier = val_brier = test_brier = float("nan")
        n_train = n_val = n_test = 0

        try:
            if oof_models:
                # Full-sample predictions from the latest window model
                d_all = xgb.DMatrix(
                    X.values,
                    feature_names=X.columns.tolist(),
                )
                proba_all_full = oof_models[-1].predict(d_all)

                proba_df_full = None
                if proba_all_full.ndim == 2 and proba_all_full.shape[1] >= 3:
                    class_to_scalar_full = np.array([0.0, 0.5, 1.0], dtype=np.float32)
                    report_y_pred_scalar = proba_all_full.dot(class_to_scalar_full)
                    report_y_pred_class = np.argmax(proba_all_full, axis=1)
                    proba_df_full = pd.DataFrame(
                        proba_all_full,
                        index=df_with_target.index,
                        columns=prob_cols,
                    )
                else:
                    report_y_pred_scalar = np.asarray(proba_all_full, dtype=float)
                    report_y_pred_class = None

                # Feature importances from the booster
                try:
                    score_dict = oof_models[-1].get_score(importance_type="gain")
                    report_feature_importances = np.array(
                        [float(score_dict.get(col, 0.0)) for col in feature_cols],
                        dtype=float,
                    )
                except Exception:
                    report_feature_importances = None

                # Train/val/test metrics on scalar predictions
                y_true_reg = df_with_target["target_atr_return"]
                target_class_series = pd.Series(target_class, index=df_with_target.index)
                y_pred_series = pd.Series(report_y_pred_scalar, index=df_with_target.index)

                def _compute_split_metrics(start_ts, end_ts):
                    mask = (df_with_target.index >= start_ts) & (df_with_target.index <= end_ts)
                    y_true_split = y_true_reg[mask]
                    y_pred_split = y_pred_series[mask]
                    if len(y_true_split) == 0:
                        return (
                            0,
                            float("nan"),
                            float("nan"),
                            float("nan"),
                        )

                    rmse_val = float(
                        np.sqrt(mean_squared_error(y_true_split, y_pred_split))
                    )
                    try:
                        r2_val = float(r2_score(y_true_split, y_pred_split))
                    except Exception:
                        r2_val = float("nan")

                    brier_val = float("nan")
                    if proba_df_full is not None:
                        proba_split = proba_df_full.loc[mask]
                        if len(proba_split) == len(y_true_split):
                            y_class_split = target_class_series[mask].to_numpy()
                            y_onehot = np.eye(3, dtype=float)[y_class_split]
                            brier_scores = []
                            for class_idx in range(3):
                                try:
                                    brier_scores.append(
                                        brier_score_loss(
                                            y_onehot[:, class_idx],
                                            proba_split.iloc[:, class_idx].to_numpy(),
                                        )
                                    )
                                except Exception:
                                    continue
                            if brier_scores:
                                brier_val = float(np.mean(brier_scores))

                    return (
                        int(mask.sum()),
                        rmse_val,
                        r2_val,
                        brier_val,
                    )

                n_train, train_rmse, train_r2, train_brier = _compute_split_metrics(
                    split_config.training.start,
                    split_config.training.effective_end,
                )
                n_val, val_rmse, val_r2, val_brier = _compute_split_metrics(
                    split_config.validation.start,
                    split_config.validation.effective_end,
                )
                n_test, test_rmse, test_r2, test_brier = _compute_split_metrics(
                    split_config.test.start,
                    split_config.test.effective_end,
                )
        except Exception as report_metric_exc:
            tprint_warning(f"SMC XGB: failed to compute full report metrics: {report_metric_exc}")

            tprint_success(
                f"✅ SMC XGB OOF metrics: accuracy={oof_accuracy:.4f}, f1={oof_f1:.4f}, "
                f"logloss={oof_logloss:.4f}, rmse={oof_rmse:.4f}, r2={oof_r2:.4f}, ic={oof_ic:.4f}"
            )
        else:
            tprint_warning("No OOF predictions available for metric calculation")

        # Generate comprehensive SMC reports if we have predictions
        try:
            if report_y_pred_scalar is not None and len(report_y_pred_scalar) == len(df_with_target):
                report_artifacts = self._generate_smc_reports(
                    df=df_with_target,
                    model=oof_models[-1] if oof_models else None,
                    X=X,
                    y=df_with_target["target_atr_return"],
                    forward_returns=df_with_target["forward_return"],
                    feature_cols=feature_cols,
                    calibration_results=None,
                    symbol=symbol,
                    exchange=exchange,
                    regime_timeframe=regime_timeframe,
                    train_rmse=train_rmse,
                    val_rmse=val_rmse,
                    test_rmse=test_rmse,
                    train_r2=train_r2,
                    val_r2=val_r2,
                    test_r2=test_r2,
                    train_brier=train_brier,
                    val_brier=val_brier,
                    test_brier=test_brier,
                    iso_test_rmse=float("nan"),
                    iso_test_r2=float("nan"),
                    n_train=n_train,
                    n_val=n_val,
                    n_test=n_test,
                    y_pred_scalar=report_y_pred_scalar,
                    y_pred_class=report_y_pred_class,
                    feature_importance_override=report_feature_importances,
                )
                if report_artifacts:
                    artifacts.extend(report_artifacts)
        except Exception as report_exc:
            tprint_warning(f"SMC XGB report generation failed: {report_exc}")

        # Save model and predictions
        model_metadata = {
            "symbol": symbol,
            "exchange": exchange,
            "timeframe": regime_timeframe,
            "n_features": len(feature_cols),
            # Use OOF predictions where available, backed off with full-sample
            # predictions from the latest model elsewhere for diagnostics.
            "prediction_method": "oof+full_merged",
            "oof_windows": len(oof_metadata),
            "hpo_runs": sum(1 for m in oof_metadata if m.get('used_hpo', False)),
            "retrain_interval_days": 10,
            "hpo_interval_days": 30,
            "training_start": str(split_config.training.start),
            "training_end": str(split_config.training.effective_end),
            "validation_start": str(split_config.validation.start),
            "validation_end": str(split_config.validation.effective_end),
            "test_start": str(split_config.test.start),
            "test_end": str(split_config.test.effective_end),
        }
        if split_config.burnin is not None:
            model_metadata["burnin_start"] = str(split_config.burnin.start)
            model_metadata["burnin_end"] = str(split_config.burnin.effective_end)

        model_path = self._save_artifact(
            data=oof_models[-1] if oof_models else None,
            artifact_name="smc_xgb_model",
            artifact_type="model",
            data_category="models",
            metadata=model_metadata,
        )
        artifacts.append(model_path)

        # ------------------------------------------------------------------
        # Persist scalar SMC predictions with full temporal coverage
        # ------------------------------------------------------------------
        # Build a scalar prediction series aligned to df_with_target.index.
        # - Where OOF predictions exist, use their scalar values.
        # - Elsewhere, use full-sample predictions from the latest model
        #   (report_y_pred_scalar) so diagnostics see the complete history.
        if report_y_pred_scalar is not None:
            try:
                full_pred_series = pd.Series(
                    report_y_pred_scalar,
                    index=df_with_target.index,
                    dtype=float,
                )

                # Overlay OOF scalar predictions where available
                if not oof_predictions.empty and "scalar" in oof_predictions.columns:
                    try:
                        full_pred_series.loc[oof_predictions.index] = (
                            oof_predictions["scalar"].astype(float)
                        )
                    except Exception:
                        # Best-effort overlay; fall back to full-sample only on error
                        pass

                # Respect burn-in: mask out any timestamps before the
                # effective burn-in end so diagnostics don't treat the
                # warm-up period as valid signal.
                try:
                    if split_config.burnin is not None and split_config.burnin.effective_end is not None:
                        burnin_cutoff = split_config.burnin.effective_end
                        full_pred_series.loc[full_pred_series.index < burnin_cutoff] = np.nan
                except Exception:
                    # Non-fatal; if burn-in metadata is unavailable, keep series as-is
                    pass

                predictions_df = full_pred_series.to_frame(name="predicted").reset_index().rename(
                    columns={full_pred_series.index.name or "index": "timestamp"}
                )
            except Exception:
                # Fallback: keep legacy OOF-only behavior
                predictions_df = oof_predictions.reset_index().rename(
                    columns={oof_predictions.index.name or "index": "timestamp"}
                )
                predictions_df = predictions_df.rename(columns={"scalar": "predicted"})
                predictions_df = predictions_df[["timestamp", "predicted"]]
        else:
            # No full-sample predictions available; fall back to OOF-only
            predictions_df = oof_predictions.reset_index().rename(
                columns={oof_predictions.index.name or "index": "timestamp"}
            )
            predictions_df = predictions_df.rename(columns={"scalar": "predicted"})
            predictions_df = predictions_df[["timestamp", "predicted"]]

        predictions_path = self._save_artifact(
            data=predictions_df,
            artifact_name="smc_predictions_with_confidence",
            artifact_type="data",
            data_category="predictions",
            metadata=model_metadata,
        )
        artifacts.append(predictions_path)

        return metrics, artifacts

    def _run_hpo(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run Bayesian TPE hyperparameter optimization for multi-class classification.
        
        Enhanced with:
        - Warm start from previous best params
        - Dynamic subsampling based on dataset size
        - Saving best params for future runs
        """
        try:
            import optuna
            from optuna.samplers import TPESampler
        except ImportError:
            tprint_warning("Optuna not available, using default params")
            return {
                "objective": "multi:softprob",
                "num_class": 3,
                "n_estimators": 500,
                "learning_rate": 0.03,
                "max_depth": 4,
                "subsample": 0.6,
                "colsample_bytree": 0.6,
                "gamma": 0.5,
                "reg_alpha": 1.0,
                "reg_lambda": 2.0,
                "min_child_weight": 15,
                "random_state": 42,
                "n_jobs": -1,
            }

        # ================================================================
        # Warm Start and Dynamic Subsampling Integration
        # ================================================================
        try:
            from src.utils.ml_common.training_efficiency import WarmStartManager, DynamicSubsampler
            
            # Setup warm start manager
            symbol = config.get('symbol', 'UNKNOWN')
            timeframe = config.get('timeframe', '15m')
            model_id = f"{symbol}_{timeframe}_smc_regime"
            warm_manager = WarmStartManager(model_id=model_id, model_type='smc_xgb')
            warm_params = warm_manager.load_params()
            
            # Dynamic subsampling for HPO (use 10-50% of data based on size)
            subsampler = DynamicSubsampler()
            n_original = len(X_train)
            X_hpo, y_hpo = subsampler.sample(X_train, y_train, stratify=True)
            sample_info = subsampler.get_subsample_info(n_original)
            
            if sample_info['will_subsample']:
                tprint_info(
                    f"🎯 Dynamic subsampling: {sample_info['original_samples']} -> "
                    f"{sample_info['sampled_size']} ({sample_info['sample_pct']:.1%}) for HPO"
                )
            
            if warm_params:
                tprint_info(f"🔄 Loaded warm start params: {list(warm_params.keys())}")
        except ImportError:
            tprint_warning("training_efficiency module not available, skipping warm start/subsampling")
            X_hpo = X_train
            y_hpo = y_train
            warm_params = None
            warm_manager = None
        except Exception as e:
            tprint_warning(f"Warm start/subsampling setup failed: {e}")
            X_hpo = X_train
            y_hpo = y_train
            warm_params = None
            warm_manager = None

        # Split for validation
        val_frac = 0.2
        split_idx = int(len(X_hpo) * (1 - val_frac))
        X_tr = X_hpo.iloc[:split_idx]
        y_tr = y_hpo.iloc[:split_idx]
        X_val = X_hpo.iloc[split_idx:]
        y_val = y_hpo.iloc[split_idx:]

        def objective(trial):
            import xgboost as xgb
            from sklearn.metrics import log_loss, f1_score

            # Emphasize directional IC more strongly by default.
            ic_weight = float(config.get("smc_hpo_ic_weight", 0.9))

            params = {
                "objective": "multi:softprob",
                "num_class": 3,
                "n_estimators": trial.suggest_int("n_estimators", 300, 900),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 7),
                "subsample": trial.suggest_float("subsample", 0.5, 0.9),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
                "gamma": trial.suggest_float("gamma", 0.1, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 3.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 3.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 5, 30),
                "random_state": 42,
                "n_jobs": -1,
            }

            try:
                model = xgb.XGBClassifier(**params)
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                proba_val = model.predict_proba(X_val)

                # Base objective: minimize log loss (multi-class)
                loss = log_loss(y_val, proba_val, labels=[0, 1, 2])

                # Lightly regularize against low F1 by adding a penalty
                y_pred_cls = np.argmax(proba_val, axis=1)
                f1 = f1_score(y_val, y_pred_cls, average="macro")
                penalty = max(0.0, 0.6 - f1)

                # Directional IC-like term on scalarized predictions vs label direction
                class_to_scalar = np.array([0.0, 0.5, 1.0], dtype=np.float32)
                pred_scalar = proba_val.dot(class_to_scalar)
                y_dir = np.where(y_val == 2, 1.0, np.where(y_val == 0, -1.0, 0.0)).astype(np.float32)

                mask = np.isfinite(pred_scalar) & np.isfinite(y_dir)
                if mask.sum() > 50:
                    try:
                        ic = float(np.corrcoef(pred_scalar[mask], y_dir[mask])[0, 1])
                    except Exception:
                        ic = 0.0
                else:
                    ic = 0.0

                # Minimize logloss + F1 penalty - directional IC term
                return loss + penalty - ic_weight * ic

            except Exception as e:
                tprint_warning(f"HPO trial failed: {e}")
                return float("inf")

        # Create and run study
        n_trials = int(config.get("smc_xgb_hpo_trials", 30))
        sampler = TPESampler(seed=42)
        study = optuna.create_study(direction='minimize', sampler=sampler)

        # Enqueue warm start params as first trial if available
        if warm_params:
            try:
                # Filter to only include HPO parameters
                hpo_param_names = ['n_estimators', 'learning_rate', 'max_depth', 'subsample', 
                                   'colsample_bytree', 'gamma', 'reg_alpha', 'reg_lambda', 'min_child_weight']
                warm_trial = {k: v for k, v in warm_params.items() if k in hpo_param_names}
                if warm_trial:
                    study.enqueue_trial(warm_trial)
                    tprint_info(f"🔄 Enqueued warm start trial with {len(warm_trial)} params")
            except Exception as e:
                tprint_warning(f"Failed to enqueue warm start trial: {e}")

        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        tprint_info(f"HPO completed: best objective value={study.best_value:.4f}")

        best_params = study.best_params
        best_params.update(
            {
                "objective": "multi:softprob",
                "num_class": 3,
                "random_state": 42,
                "n_jobs": -1,
            }
        )

        # Save best params for future warm start
        if warm_manager is not None:
            try:
                warm_manager.save_params(best_params, metrics={'best_value': study.best_value})
                tprint_info("💾 Saved best params for future warm start")
            except Exception as e:
                tprint_warning(f"Failed to save warm start params: {e}")

        return best_params

    def _calibrate_conformal_prediction(
        self,
        model,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Optional[Dict[str, Any]]:
        """Perform conformal prediction calibration with extended confidence levels."""
        try:
            cal_frac = 0.2
            split_idx = int(len(X_train) * (1 - cal_frac))

            X_cal = X_train.iloc[split_idx:]
            y_cal = y_train.iloc[split_idx:]

            if len(X_cal) == 0:
                return None

            y_cal_pred = model.predict(X_cal)
            nonconformity_scores = np.abs(y_cal - y_cal_pred)

            confidence_levels = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
            quantiles = {}
            coverage_metrics = {}

            for alpha in confidence_levels:
                q = np.percentile(nonconformity_scores, alpha * 100)
                key = f"{int(alpha * 100)}%"
                quantiles[key] = float(q)
                coverage = float((nonconformity_scores <= q).mean())
                coverage_metrics[f"conformal_coverage_{key}"] = coverage

            tprint_info(f"Conformal prediction quantiles: {quantiles}")

            all_X = pd.concat([X_train, X_test])
            all_preds = model.predict(all_X)

            confidence_scores = {}
            for level, q in quantiles.items():
                conf_score = q / (np.abs(all_preds - 0.5) + q + 1e-9)
                confidence_scores[level] = conf_score.clip(0.0, 1.0)

            metrics_dict = {
                "conformal_quantile_50": quantiles.get("50%", float("nan")),
                "conformal_quantile_70": quantiles.get("70%", float("nan")),
                "conformal_quantile_90": quantiles.get("90%", float("nan")),
            }
            metrics_dict.update(coverage_metrics)

            return {
                "calibration": {
                    "quantiles": quantiles,
                    "nonconformity_scores": nonconformity_scores.tolist(),
                },
                "confidence_scores": confidence_scores,
                "metrics": metrics_dict,
            }

        except Exception as e:
            tprint_warning(f"Conformal calibration failed: {e}")
            return None

    def _generate_smc_reports(
        self,
        df: pd.DataFrame,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        forward_returns: pd.Series,
        feature_cols: List[str],
        calibration_results: Optional[Dict[str, Any]],
        symbol: str,
        exchange: str,
        regime_timeframe: str,
        train_rmse: float,
        val_rmse: float,
        test_rmse: float,
        train_r2: float,
        val_r2: float,
        test_r2: float,
        train_brier: float,
        val_brier: float,
        test_brier: float,
        iso_test_rmse: float,
        iso_test_r2: float,
        n_train: int,
        n_val: int,
        n_test: int,
        y_pred_scalar: Optional[np.ndarray] = None,
        y_pred_class: Optional[np.ndarray] = None,
        feature_importance_override: Optional[np.ndarray] = None,
    ) -> List[str]:
        """Generate consolidated comprehensive reports."""
        artifacts = []

        out_dir = Path("outcomes")
        out_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Predictions (scalarized classifier output 0.0/0.5/1.0 where possible)
        if y_pred_scalar is not None:
            y_pred_scalar = np.asarray(y_pred_scalar, dtype=float)
            if y_pred_class is not None:
                y_pred_class = np.asarray(y_pred_class, dtype=int)
        else:
            try:
                proba_all = model.predict_proba(X)
                if proba_all.shape[1] >= 3:
                    class_to_scalar = np.array([0.0, 0.5, 1.0], dtype=np.float32)
                    y_pred_scalar = proba_all.dot(class_to_scalar)
                    y_pred_class = np.argmax(proba_all, axis=1)
                else:
                    y_pred_scalar = model.predict(X)
                    y_pred_class = None
            except Exception:
                y_pred_scalar = model.predict(X)
                y_pred_class = None

        # 1. CONSOLIDATED CSV: Feature importance + correlations
        if feature_importance_override is not None:
            feature_importance = np.asarray(feature_importance_override, dtype=float)
        else:
            feature_importance = getattr(model, "feature_importances_", None)
            if feature_importance is None:
                feature_importance = np.zeros(len(feature_cols), dtype=float)

        feature_data = []
        for i, feat in enumerate(feature_cols):
            corr = X[feat].corr(y) if feat in X.columns else 0.0
            feature_data.append({
                "feature": feat,
                "importance": float(feature_importance[i]) if i < len(feature_importance) else 0.0,
                "correlation_with_target": float(corr) if not np.isnan(corr) else 0.0,
            })

        features_df = pd.DataFrame(feature_data).sort_values("importance", ascending=False)

        features_csv = out_dir / f"smc_features_analysis_{symbol}_{regime_timeframe}_{ts}.csv"
        features_df.to_csv(features_csv, index=False)
        artifacts.append(str(features_csv))

        # 2. COMPREHENSIVE MARKDOWN REPORT with integrated distributions
        md_lines = []
        md_lines.append("# SMC XGBoost Model Report")
        md_lines.append("")
        md_lines.append(f"- **Symbol**: {symbol}")
        md_lines.append(f"- **Exchange**: {exchange}")
        md_lines.append(f"- **Timeframe**: {regime_timeframe}")
        md_lines.append(f"- **Generated**: {ts}")
        md_lines.append("")

        # Model Performance
        md_lines.append("## Model Performance")
        md_lines.append("")
        md_lines.append("| Metric | Train | Val | Test |")
        md_lines.append("| --- | --- | --- | --- |")
        md_lines.append(
            f"| RMSE | {train_rmse:.4f} | {val_rmse:.4f} | {test_rmse:.4f} |"
        )
        md_lines.append(
            f"| R² Score | {train_r2:.4f} | {val_r2:.4f} | {test_r2:.4f} |"
        )
        md_lines.append(
            f"| Brier Score | {train_brier:.4f} | {val_brier:.4f} | {test_brier:.4f} |"
        )
        md_lines.append(
            f"| Samples | {n_train} | {n_val} | {n_test} |"
        )
        md_lines.append(f"| Features | {len(feature_cols)} | {len(feature_cols)} | {len(feature_cols)} |")
        md_lines.append("")

        if np.isfinite(iso_test_rmse) or np.isfinite(iso_test_r2):
            md_lines.append("### Isotonic-Calibrated Performance (Test)")
            md_lines.append("")
            md_lines.append("| Metric | Value |")
            md_lines.append("| --- | --- |")
            md_lines.append(f"| Iso RMSE | {iso_test_rmse:.4f} |")
            md_lines.append(f"| Iso R² | {iso_test_r2:.4f} |")
            md_lines.append("")

        # Mean returns by signal
        if y_pred_class is not None:
            breakout_pred = y_pred_class == 2
            breakdown_pred = y_pred_class == 0
        else:
            breakout_pred = y_pred_scalar > 0.75
            breakdown_pred = y_pred_scalar < 0.25
        if breakout_pred.sum() > 0:
            mean_ret_breakout = forward_returns[breakout_pred].mean()
            md_lines.append(f"**Mean Return (Breakout Signals)**: {mean_ret_breakout:.4f} ({breakout_pred.sum()} signals)")
        if breakdown_pred.sum() > 0:
            mean_ret_breakdown = forward_returns[breakdown_pred].mean()
            md_lines.append(f"**Mean Return (Breakdown Signals)**: {mean_ret_breakdown:.4f} ({breakdown_pred.sum()} signals)")
        md_lines.append("")

        # Top features
        md_lines.append("## Top 15 Features by Importance")
        md_lines.append("")
        md_lines.append("| Rank | Feature | Importance | Correlation |")
        md_lines.append("| --- | --- | --- | --- |")
        for i, row in features_df.head(15).iterrows():
            md_lines.append(f"| {i+1} | {row['feature']} | {row['importance']:.4f} | {row['correlation_with_target']:.4f} |")
        md_lines.append("")

        # Conformal prediction
        if calibration_results:
            md_lines.append("## Conformal Prediction Calibration")
            md_lines.append("")
            md_lines.append("Prediction intervals for uncertainty quantification:")
            md_lines.append("")
            md_lines.append("| Confidence Level | Quantile |")
            md_lines.append("| --- | --- |")
            for level, q in calibration_results["calibration"]["quantiles"].items():
                md_lines.append(f"| {level} | ±{q:.4f} |")
            md_lines.append("")

        # Prediction distribution (integrated)
        md_lines.append("## Prediction Distribution Analysis")
        md_lines.append("")
        md_lines.append("| Statistic | Predicted | Actual |")
        md_lines.append("| --- | --- | --- |")
        md_lines.append(f"| Mean | {y_pred_scalar.mean():.4f} | {y.mean():.4f} |")
        md_lines.append(f"| Std | {y_pred_scalar.std():.4f} | {y.std():.4f} |")
        md_lines.append(f"| Min | {y_pred_scalar.min():.4f} | {y.min():.4f} |")
        md_lines.append(
            f"| 25th Percentile | {np.percentile(y_pred_scalar, 25):.4f} | {np.percentile(y, 25):.4f} |"
        )
        md_lines.append(f"| Median | {np.median(y_pred_scalar):.4f} | {np.median(y):.4f} |")
        md_lines.append(
            f"| 75th Percentile | {np.percentile(y_pred_scalar, 75):.4f} | {np.percentile(y, 75):.4f} |"
        )
        md_lines.append(f"| Max | {y_pred_scalar.max():.4f} | {y.max():.4f} |")
        md_lines.append("")

        # Scalar band performance (deciles of scalar prediction)
        md_lines.append("## Scalar Band Performance (Deciles)")
        md_lines.append("")
        try:
            arr_pred = np.asarray(y_pred_scalar, dtype=float)
            arr_ret = np.asarray(forward_returns.to_numpy(), dtype=float)
            mask = np.isfinite(arr_pred) & np.isfinite(arr_ret)
            if mask.sum() >= 100:
                band_df = pd.DataFrame({"pred": arr_pred[mask], "ret": arr_ret[mask]})
                band_df["band"] = pd.qcut(band_df["pred"], q=10, duplicates="drop")

                def _band_stats(g: pd.DataFrame) -> pd.Series:
                    pred_vals = g["pred"].to_numpy()
                    ret_vals = g["ret"].to_numpy()
                    direction = np.sign(pred_vals - 0.5)
                    ret_sign = np.sign(ret_vals)
                    dir_mask = direction != 0
                    if dir_mask.any():
                        hit_rate = float(
                            (direction[dir_mask] * ret_sign[dir_mask] > 0).mean()
                        )
                    else:
                        hit_rate = float("nan")
                    return pd.Series(
                        {
                            "count": int(len(g)),
                            "pred_mean": float(pred_vals.mean()),
                            "ret_mean": float(ret_vals.mean()),
                            "hit_rate": hit_rate,
                        }
                    )

                stats = band_df.groupby("band").apply(_band_stats)

                md_lines.append(
                    "| Band | Count | Mean Scalar | Mean Forward Return | Directional Hit Rate |"
                )
                md_lines.append("| --- | --- | --- | --- | --- |")
                for band_label, row in stats.iterrows():
                    md_lines.append(
                        f"| {band_label} | {int(row['count'])} | "
                        f"{row['pred_mean']:.4f} | {row['ret_mean']:.4f} | "
                        f"{row['hit_rate']:.2%} |"
                    )
                md_lines.append("")
            else:
                md_lines.append("_Not enough samples for scalar band analysis._")
                md_lines.append("")
        except Exception:
            md_lines.append("_Scalar band performance analysis failed; see logs for details._")
            md_lines.append("")

        # Directional accuracy
        breakout_actual = y > 1.0
        breakdown_actual = y < 0.0
        if breakout_pred.sum() > 0:
            breakout_acc = (breakout_actual[breakout_pred]).mean()
            md_lines.append(f"**Breakout Prediction Accuracy**: {breakout_acc:.2%} ({breakout_pred.sum()} predictions)")
        if breakdown_pred.sum() > 0:
            breakdown_acc = (breakdown_actual[breakdown_pred]).mean()
            md_lines.append(f"**Breakdown Prediction Accuracy**: {breakdown_acc:.2%} ({breakdown_pred.sum()} predictions)")
        md_lines.append("")

        # Confidence analysis (integrated)
        if calibration_results:
            md_lines.append("## Confidence Score Analysis")
            md_lines.append("")
            md_lines.append("| Confidence Level | Mean Score | Std Score |")
            md_lines.append("| --- | --- | --- |")
            for level, scores in calibration_results["confidence_scores"].items():
                mean_conf = np.mean(scores)
                std_conf = np.std(scores)
                md_lines.append(f"| {level} | {mean_conf:.4f} | {std_conf:.4f} |")
            md_lines.append("")

        md_path = out_dir / f"smc_xgb_report_{symbol}_{regime_timeframe}_{ts}.md"
        md_path.write_text("\n".join(md_lines), encoding="utf-8")
        artifacts.append(str(md_path))

        tprint_success(f"✅ Generated {len(artifacts)} report artifacts")

        return artifacts
