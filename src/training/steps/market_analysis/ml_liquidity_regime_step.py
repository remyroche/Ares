"""
ML Liquidity Regime Step

This step constructs liquidity-based regimes from 1h OHLCV data, focused on
Effort vs Result of price moves and participation (volume context).

Primary goals:
- Detect Valid Trend, Absorption, Ghost/Drift, and Apathy regimes.
- Train an XGBClassifier to predict regimes from liquidity and microstructure features.
- Calibrate probabilities and expose `liquidity_regime_prob_ghost` as a
  downstream feature.
- Save 1h training artifacts (model, feature pipeline, regime stats, thresholds,
  quality metrics).
- Map 1h regime probabilities down to a 15m grid and save as a dedicated
  artifact for downstream consumers.
"""

import logging
import time
from typing import Any, Dict, Optional, Tuple, List
from dataclasses import asdict, is_dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from src.training.steps.base_step import BaseStep
from src.utils.tprint import (
    tprint,
    tprint_info,
    tprint_warning,
    tprint_error,
)
from src.features_common.transforms.scaling_normalization import ScalingNormalizer
from src.training.steps.market_analysis.clusters.cluster_quality_assessor import (
    ClusterQualityAssessor,
    ClusterQualityMetrics,
)
from src.training.steps.market_analysis.clusters.liquidity_cluster_quality_assessor import (
    LiquidityClusterQualityAssessor,
    LiquidityClusterQualityMetrics,
)
from src.utils.ml_common.feature_engineering.feature_smoothing import apply_ewm_smoothing

logger = logging.getLogger(__name__)


class MLLiquidityRegimeStep(BaseStep):
    """Pipeline step to construct liquidity-based regimes from 1h OHLCV."""

    def __init__(self, step_name: str = "ml_liquidity_regime_step"):
        super().__init__(step_name, use_versioned_artifacts=True)
        self.logger = logger.getChild("MLLiquidityRegimeStep") if hasattr(logger, "getChild") else logger
        tprint(f"✅ Initialized {step_name} step", "SUCCESS")

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the liquidity regime construction and model training.

        Expected config keys (minimum):
            - symbol: Trading symbol (e.g., 'ETHUSDT')
            - exchange: Exchange name (e.g., 'binance')
            - regime_timeframe: Timeframe used for liquidity regimes (default: '1h')
            - direction: Trading direction (default: 'long')
            - execution_mode: 'full', 'light', 'blank', etc.
        """
        start_time = time.time()

        try:
            symbol = str(config.get("symbol", "ETHUSDT"))
            exchange = str(config.get("exchange", "binance"))
            regime_timeframe = str(config.get("regime_timeframe", config.get("timeframe", "1h")))
            direction = str(config.get("direction", "long"))

            if not symbol or not exchange:
                raise ValueError("Config must include 'symbol' and 'exchange'")

            # ------------------------------------------------------------------
            # 0) Default liquidity-specific configuration
            # ------------------------------------------------------------------
            liquidity_defaults: Dict[str, Any] = {
                "liquidity_output_timeframe": "15m",
                "liquidity_prob_interpolation_mode": "step",  # 'step' or 'linear'
                "liquidity_min_samples": 200,
                "liquidity_train_fraction": 0.8,
                "liquidity_use_ewm_features": True,
                "liquidity_enable_prob_calibration": True,
                "liquidity_enable_hpo": False,
                "liquidity_n_regimes": 4,
            }
            for k, v in liquidity_defaults.items():
                config.setdefault(k, v)

            tprint_info(
                f"🚀 Starting {self.step_name} for {symbol} on {exchange} "
                f"(regime_timeframe={regime_timeframe})"
            )

            # ------------------------------------------------------------------
            # 1) Load 1h OHLCV market data
            # ------------------------------------------------------------------
            self.set_context(
                symbol=symbol,
                exchange=exchange,
                timeframe=regime_timeframe,
                direction=direction,
                model="liquidity_regime",
            )

            market_data_1h, market_source = self.load_market_data_or_fail(
                {**config, "timeframe": regime_timeframe},
                pipeline_state={},
                allow_config_override=True,
            )

            if not isinstance(market_data_1h, pd.DataFrame) or market_data_1h.empty:
                raise ValueError("Loaded market data is empty or not a DataFrame")

            if not isinstance(market_data_1h.index, pd.DatetimeIndex):
                try:
                    market_data_1h = market_data_1h.copy()
                    market_data_1h.index = pd.to_datetime(market_data_1h.index)
                except Exception as exc:
                    raise ValueError("Market data index could not be converted to DatetimeIndex") from exc

            tprint_info(
                f"✅ Loaded 1h market data from {market_source}: {market_data_1h.shape} "
                f"({market_data_1h.index.min()} → {market_data_1h.index.max()})"
            )

            # ------------------------------------------------------------------
            # 2) Generate liquidity features on 1h grid
            # ------------------------------------------------------------------
            liquidity_df = self._generate_liquidity_features(market_data_1h, config)

            # ------------------------------------------------------------------
            # 3) Construct semantic liquidity regimes (0–3)
            # ------------------------------------------------------------------
            liquidity_df = self._assign_liquidity_regimes(liquidity_df, config)

            # ------------------------------------------------------------------
            # 4) Train XGBClassifier on liquidity regimes
            # ------------------------------------------------------------------
            (
                model,
                proba_df,
                regime_labels,
                training_metrics,
                feature_pipeline_artifacts,
            ) = self._train_liquidity_model(liquidity_df, config)

            # Attach probabilities & core feature for downstream
            if proba_df is not None:
                for col in proba_df.columns:
                    liquidity_df[col] = proba_df[col]
                if "liquidity_regime_prob_ghost" not in liquidity_df.columns and "p_regime_3" in proba_df.columns:
                    liquidity_df["liquidity_regime_prob_ghost"] = proba_df["p_regime_3"]

            # ------------------------------------------------------------------
            # 5) Assess regime quality
            #   a) Generic cluster quality via ClusterQualityAssessor
            #   b) Liquidity-specific quality via LiquidityClusterQualityAssessor
            # ------------------------------------------------------------------
            liquidity_quality_metrics: Optional[ClusterQualityMetrics] = None
            liquidity_quality_path: Optional[str] = None

            try:
                liquidity_quality_metrics, liquidity_quality_path = self._assess_liquidity_regime_quality(
                    liquidity_df=liquidity_df,
                    regime_col="liquidity_regime",
                    config=config,
                )
            except Exception as quality_exc:
                tprint_warning(f"Liquidity regime quality assessment failed: {quality_exc}")

            # Liquidity-specific quality assessment & reports
            liquidity_cluster_metrics: Optional[LiquidityClusterQualityMetrics] = None
            liquidity_cluster_md_path: Optional[str] = None
            liquidity_cluster_csv_path: Optional[str] = None
            liquidity_cluster_metrics_path: Optional[str] = None

            try:
                # 1h forward returns as secondary diagnostic
                forward_returns_1h = liquidity_df.get("return_1h")
                if forward_returns_1h is not None:
                    forward_returns_1h = forward_returns_1h.shift(-1)

                assessor = LiquidityClusterQualityAssessor(config=config)
                liquidity_cluster_metrics = assessor.assess_liquidity_clusters(
                    liquidity_df=liquidity_df,
                    regime_labels=liquidity_df["liquidity_regime"].astype(int),
                    forward_returns_1h=forward_returns_1h,
                    config=config,
                )

                # Expose CoV-based and overall quality metrics for multi-criteria selection
                training_metrics["liquidity_effort_result_cov_separation_score"] = float(
                    liquidity_cluster_metrics.effort_result_cov_separation_score
                )
                training_metrics["liquidity_returns_cov_separation_score"] = float(
                    liquidity_cluster_metrics.returns_cov_separation_score
                )
                training_metrics["liquidity_overall_quality_score"] = float(
                    liquidity_cluster_metrics.overall_quality_score
                )

                # Generate human-readable reports in outcomes/
                liquidity_cluster_md_path = assessor.save_markdown_report(
                    metrics=liquidity_cluster_metrics,
                    symbol=symbol,
                    output_dir="outcomes",
                )
                liquidity_cluster_csv_path = assessor.save_csv_report(
                    metrics=liquidity_cluster_metrics,
                    symbol=symbol,
                    output_dir="outcomes",
                )

                # Persist metrics as versioned artifact
                try:
                    metrics_dict = {
                        "effort_result_separation_score": liquidity_cluster_metrics.effort_result_separation_score,
                        "ghost_vs_valid_contrast": liquidity_cluster_metrics.ghost_vs_valid_contrast,
                        "absorption_vs_valid_contrast": liquidity_cluster_metrics.absorption_vs_valid_contrast,
                        "effort_result_cov_separation_score": liquidity_cluster_metrics.effort_result_cov_separation_score,
                        "returns_cov_separation_score": liquidity_cluster_metrics.returns_cov_separation_score,
                        "ghost_reversal_rate": liquidity_cluster_metrics.ghost_reversal_rate,
                        "ghost_false_trend_rate": liquidity_cluster_metrics.ghost_false_trend_rate,
                        "absorption_reversal_rate": liquidity_cluster_metrics.absorption_reversal_rate,
                        "absorption_follow_through_rate": liquidity_cluster_metrics.absorption_follow_through_rate,
                        "valid_trend_follow_through": liquidity_cluster_metrics.valid_trend_follow_through,
                        "apathy_noise_fraction": liquidity_cluster_metrics.apathy_noise_fraction,
                        "class_balance_score": liquidity_cluster_metrics.class_balance_score,
                        "n_regimes": liquidity_cluster_metrics.n_regimes,
                        "n_samples": liquidity_cluster_metrics.n_samples,
                        "per_regime_metrics": liquidity_cluster_metrics.per_regime_metrics,
                        "overall_quality_score": liquidity_cluster_metrics.overall_quality_score,
                        "assessment_timestamp": liquidity_cluster_metrics.assessment_timestamp,
                    }

                    liquidity_cluster_metrics_path = self._save_artifact(
                        data=metrics_dict,
                        artifact_name="ml_liquidity_cluster_quality_metrics_1h",
                        artifact_type="data",
                        metadata={
                            "overall_quality_score": liquidity_cluster_metrics.overall_quality_score,
                            "n_regimes": liquidity_cluster_metrics.n_regimes,
                            "assessment_timestamp": liquidity_cluster_metrics.assessment_timestamp,
                        },
                    )
                except Exception as save_metrics_exc:
                    tprint_warning(
                        f"Failed to save liquidity cluster quality metrics artifact: {save_metrics_exc}"
                    )
            except Exception as liquidity_cluster_exc:
                tprint_warning(f"Liquidity-specific cluster quality assessment failed: {liquidity_cluster_exc}")

            # ------------------------------------------------------------------
            # 6) Save 1h training artifacts
            # ------------------------------------------------------------------
            liquidity_to_save = liquidity_df.reset_index().rename(
                columns={liquidity_df.index.name or "index": "timestamp"}
            )

            tprint_info(
                f"💾 Saving liquidity training dataset with shape {liquidity_to_save.shape} "
                f"to versioned HDF5 store"
            )
            training_data_path = self._save_artifact(
                data=liquidity_to_save,
                artifact_name="ml_liquidity_training_data_1h",
                artifact_type="data",
                metadata={
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": regime_timeframe,
                    "source_market_data": market_source,
                },
            )

            model_path: Optional[str] = None
            feature_pipeline_path: Optional[str] = None

            if model is not None:
                try:
                    tprint_info("💾 Saving XGBoost liquidity model via artifact router")
                    model_path = self._save_artifact(
                        data=model,
                        artifact_name="ml_liquidity_model_1h",
                        artifact_type="model",
                        metadata={
                            "symbol": symbol,
                            "exchange": exchange,
                            "timeframe": regime_timeframe,
                            "model_type": "xgboost",
                        },
                    )
                except Exception as save_model_exc:
                    tprint_warning(f"Failed to save liquidity model artifact: {save_model_exc}")

            if feature_pipeline_artifacts is not None:
                try:
                    feature_pipeline_path = self._save_artifact(
                        data=feature_pipeline_artifacts,
                        artifact_name="ml_liquidity_feature_pipeline_1h",
                        artifact_type="model",
                        metadata={
                            "symbol": symbol,
                            "exchange": exchange,
                            "timeframe": regime_timeframe,
                            "feature_names": feature_pipeline_artifacts.get("feature_names", []),
                        },
                    )
                except Exception as save_fp_exc:
                    tprint_warning(f"Failed to save liquidity feature pipeline artifact: {save_fp_exc}")

            # ------------------------------------------------------------------
            # 7) Map probabilities to 15m and save
            # ------------------------------------------------------------------
            probs_15m_path: Optional[str] = None
            try:
                probs_15m_df = self._map_probabilities_to_15m(
                    proba_df=proba_df,
                    market_data_1h=market_data_1h,
                    symbol=symbol,
                    exchange=exchange,
                    direction=direction,
                    config=config,
                )
                if probs_15m_df is not None and not probs_15m_df.empty:
                    self.set_context(
                        symbol=symbol,
                        exchange=exchange,
                        timeframe=str(config.get("liquidity_output_timeframe", "15m")),
                        direction=direction,
                        model="liquidity_regime",
                    )
                    probs_15m_to_save = probs_15m_df.reset_index().rename(
                        columns={probs_15m_df.index.name or "index": "timestamp"}
                    )
                    probs_15m_path = self._save_artifact(
                        data=probs_15m_to_save,
                        artifact_name="ml_liquidity_regime_probs_15m",
                        artifact_type="data",
                        metadata={
                            "symbol": symbol,
                            "exchange": exchange,
                            "timeframe": str(config.get("liquidity_output_timeframe", "15m")),
                        },
                    )
            except Exception as map_exc:
                tprint_warning(f"Failed to map liquidity probabilities to 15m: {map_exc}")

            execution_time = time.time() - start_time
            tprint_info(
                f"✅ {self.step_name} completed in {execution_time:.2f}s "
                f"with {len(liquidity_df)} samples"
            )

            return {
                "success": True,
                "artifacts": {
                    "liquidity_training_data": liquidity_df,
                    "liquidity_training_data_path": training_data_path,
                    "liquidity_model_path": model_path,
                    "liquidity_feature_pipeline": feature_pipeline_artifacts,
                    "liquidity_feature_pipeline_path": feature_pipeline_path,
                    "liquidity_quality_metrics": liquidity_quality_metrics,
                    "liquidity_quality_path": liquidity_quality_path,
                    "liquidity_probs_15m_path": probs_15m_path,
                },
                "metrics": training_metrics,
                "execution_time": execution_time,
            }

        except Exception as exc:
            execution_time = time.time() - start_time
            error_msg = f"{self.step_name} failed: {exc}"
            self.logger.error(error_msg, exc_info=True)
            tprint_error(error_msg)
            return {
                "success": False,
                "artifacts": {},
                "metrics": {},
                "error": str(exc),
                "execution_time": execution_time,
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _generate_liquidity_features(self, market_data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        df = market_data.copy()

        required_cols = {"open", "high", "low", "close", "volume"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required OHLCV columns for liquidity features: {missing}")

        eps = 1e-9

        # Basic derived quantities
        df["range"] = (df["high"] - df["low"]).astype(float)
        df["range"] = df["range"].replace(0, np.nan)
        df["return_1h"] = np.log(df["close"] / df["close"].shift(1)).astype(float)
        df["abs_return_1h"] = df["return_1h"].abs()
        df["dollar_volume"] = (df["close"] * df["volume"]).astype(float)

        # Relative volume context
        vol_window_daily = int(config.get("liquidity_rvol_lookback_24", 24))
        vol_window_weekly = int(config.get("liquidity_rvol_lookback_168", 168))

        df["vol_sma_24"] = df["volume"].rolling(vol_window_daily, min_periods=5).mean()
        df["vol_sma_168"] = df["volume"].rolling(vol_window_weekly, min_periods=20).mean()
        df["rvol_24"] = df["volume"] / (df["vol_sma_24"] + eps)
        df["rvol_168"] = df["volume"] / (df["vol_sma_168"] + eps)

        vol_mean_24 = df["volume"].rolling(vol_window_daily, min_periods=5).mean()
        vol_std_24 = df["volume"].rolling(vol_window_daily, min_periods=5).std()
        df["vol_z_24"] = (df["volume"] - vol_mean_24) / (vol_std_24.replace(0, np.nan) + eps)

        df["volume_stddev_stability"] = (
            df["volume"].rolling(6, min_periods=3).std() /
            (df["volume"].rolling(6, min_periods=3).mean() + eps)
        )

        # Additional stability features for regime contrast
        df["range_stddev_stability"] = (
            df["range"].rolling(6, min_periods=3).std() /
            (df["range"].rolling(6, min_periods=3).mean() + eps)
        )
        df["return_stddev_stability"] = (
            df["abs_return_1h"].rolling(6, min_periods=3).std() /
            (df["abs_return_1h"].rolling(6, min_periods=3).mean() + eps)
        )

        # Normalized range (Effort)
        range_std_lookback = int(config.get("liquidity_range_std_lookback", 48))
        range_std = df["range"].rolling(range_std_lookback, min_periods=10).std()
        df["normalized_range"] = df["range"] / (range_std.replace(0, np.nan) + eps)

        # Effort vs Result ratios
        df["normalized_volume"] = np.log1p(df["volume"])  # log volume
        df["ghost_ratio"] = df["normalized_range"] / (df["normalized_volume"] + eps)
        df["absorption_ratio"] = df["normalized_volume"] / (df["normalized_range"] + eps)

        # Amihud / Amivest
        df["amihud_validity"] = df["abs_return_1h"] / (df["dollar_volume"] + eps)
        df["amivest_efficiency"] = df["dollar_volume"] / (df["abs_return_1h"] + eps)

        # Ease of Movement (EMV)
        mid_price = (df["high"] + df["low"]) / 2.0
        mid_price_prev = mid_price.shift(1)
        df["emv"] = (mid_price - mid_price_prev) / ((df["volume"] / (df["range"] + eps)) + eps)

        # Candle geometry
        df["clv"] = (df["close"] - df["low"]) / (df["range"] + eps) - 0.5
        upper_wick = (df["high"] - df[["close", "open"]].max(axis=1)).clip(lower=0)
        lower_wick = (df[["close", "open"]].min(axis=1) - df["low"]).clip(lower=0)
        df["wick_ratio"] = np.maximum(upper_wick, lower_wick) / (df["range"] + eps)
        df["body_dominance"] = (df["close"] - df["open"]) / (df["range"] + eps)
        df["gap_factor"] = (df["open"] - df["close"].shift(1)) / (df["close"].shift(1) + eps)

        # Intraday vs closing volatility feature
        df["intraday_close_ratio"] = df["range"] / (df["abs_return_1h"].replace(0, np.nan) + eps)

        # Winsorize heavy tails for stability
        winsor_lower = float(config.get("liquidity_winsor_lower", 0.005))
        winsor_upper = float(config.get("liquidity_winsor_upper", 0.995))
        winsor_cols = [
            "rvol_24",
            "rvol_168",
            "vol_z_24",
            "normalized_range",
            "ghost_ratio",
            "absorption_ratio",
            "amihud_validity",
            "amivest_efficiency",
            "emv",
            "intraday_close_ratio",
        ]
        for col in winsor_cols:
            if col in df.columns:
                series = df[col].dropna()
                if len(series) == 0:
                    continue
                lo = series.quantile(winsor_lower)
                hi = series.quantile(winsor_upper)
                df[col] = df[col].clip(lower=lo, upper=hi)

        # ============================================================================
        # LIQUIDITY REGIME FEATURES: Comprehensive Multi-Category Implementation
        # Timeframes aligned to 30m-3h trading duration:
        # - 1h: Immediate price action (current bar context)
        # - 3h: Trade-matched window (max 3h trade duration)
        # - 6h: Structural context (2× longest trade, intermediate regime)
        # ============================================================================

        # ============================================================================
        # CATEGORY 1: Directional Orderflow Features (OHLCV-only approximation)
        # ============================================================================
        # Approximate bid-ask imbalance using close position in range

        # Close Location in Range: 0 = closed at low, 1 = closed at high
        df["close_position_range"] = (df["close"] - df["low"]) / (df["range"] + eps)
        df["close_position_range"] = df["close_position_range"].clip(0, 1)

        # Volume-weighted directional intensity
        df["volume_buyer_intensity"] = df["volume"] * df["close_position_range"]
        df["volume_seller_intensity"] = df["volume"] * (1.0 - df["close_position_range"])

        # Directional volume imbalance: [-1, 1] where 1 = all buying, -1 = all selling
        total_dir_volume = df["volume_buyer_intensity"] + df["volume_seller_intensity"]
        df["volume_direction_imbalance"] = (
            (df["volume_buyer_intensity"] - df["volume_seller_intensity"]) / (total_dir_volume + eps)
        )

        # Directional conviction: strength of one-sided flow (higher = trend, lower = apathy)
        df["volume_direction_conviction"] = df["volume_direction_imbalance"].abs()

        # Order flow persistence: does direction persist across bars?
        df["direction_change"] = (
            (df["close"] > df["close"].shift(1)).astype(float) -
            (df["close"] < df["close"].shift(1)).astype(float)
        )
        df["volume_direction_consistency"] = (
            df["volume_direction_imbalance"] * df["direction_change"]
        )

        # Smooth orderflow metrics with 3-bar and 6-bar EWMA
        if config.get("liquidity_use_ewm_features", True):
            df["volume_direction_imbalance_ewm3"] = (
                df["volume_direction_imbalance"].ewm(span=3, adjust=False).mean()
            )
            df["volume_direction_imbalance_ewm6"] = (
                df["volume_direction_imbalance"].ewm(span=6, adjust=False).mean()
            )
            df["volume_direction_conviction_ewm3"] = (
                df["volume_direction_conviction"].ewm(span=3, adjust=False).mean()
            )
            df["volume_direction_conviction_ewm6"] = (
                df["volume_direction_conviction"].ewm(span=6, adjust=False).mean()
            )

        # ============================================================================
        # CATEGORY 2: Trend Persistence Metrics (Directional Autocorrelation & Momentum)
        # ============================================================================
        # Consecutive bars in same direction (3-bar and 6-bar windows)
        direction_sign = np.sign(df["return_1h"])
        direction_sign = direction_sign.replace(0, np.nan)

        same_direction_3 = (direction_sign == direction_sign.shift(1)).astype(float)
        df["consecutive_direction_bars_3h"] = same_direction_3.rolling(window=3, min_periods=1).sum()
        df["consecutive_direction_ratio_3h"] = df["consecutive_direction_bars_3h"] / 3.0

        same_direction_6 = (direction_sign == direction_sign.shift(1)).astype(float)
        df["consecutive_direction_bars_6h"] = same_direction_6.rolling(window=6, min_periods=1).sum()
        df["consecutive_direction_ratio_6h"] = df["consecutive_direction_bars_6h"] / 6.0

        # Directional autocorrelation (lag-1): do moves persist?
        df["return_autocorr_lag1_3h"] = df["return_1h"].rolling(window=3, min_periods=2).apply(
            lambda x: x.iloc[-1] * x.iloc[-2] if len(x) >= 2 else 0, raw=False
        )
        df["return_autocorr_lag1_6h"] = df["return_1h"].rolling(window=6, min_periods=2).apply(
            lambda x: x.iloc[-1] * x.iloc[-2] if len(x) >= 2 else 0, raw=False
        )

        # Momentum: price move × volume (directional conviction)
        df["momentum_1h"] = df["return_1h"] * df["volume"]

        # Momentum persistence: does momentum amplify or decay?
        momentum_ma_3 = df["return_1h"].rolling(window=3, min_periods=2).mean()
        momentum_ma_6 = df["return_1h"].rolling(window=6, min_periods=2).mean()
        df["momentum_persistence_3h"] = (momentum_ma_3 - momentum_ma_6) / (abs(momentum_ma_6) + eps)

        # Momentum direction alignment with volume conviction
        df["momentum_volume_alignment"] = (
            np.sign(df["return_1h"]) * df["volume_direction_conviction"]
        )

        # Trend confirmation: sustained directional move with supporting volume
        df["trend_confirmation_3h"] = (
            df["consecutive_direction_ratio_3h"] * df["volume_direction_conviction"]
        )
        df["trend_confirmation_6h"] = (
            df["consecutive_direction_ratio_6h"] * df["volume_direction_conviction"]
        )

        # Smooth persistence metrics
        if config.get("liquidity_use_ewm_features", True):
            df["momentum_persistence_3h_ewm3"] = (
                df["momentum_persistence_3h"].ewm(span=3, adjust=False).mean()
            )
            df["momentum_persistence_3h_ewm6"] = (
                df["momentum_persistence_3h"].ewm(span=6, adjust=False).mean()
            )
            df["trend_confirmation_3h_ewm3"] = (
                df["trend_confirmation_3h"].ewm(span=3, adjust=False).mean()
            )
            df["trend_confirmation_6h_ewm3"] = (
                df["trend_confirmation_6h"].ewm(span=3, adjust=False).mean()
            )

        # ============================================================================
        # CATEGORY 3: Volatility-Momentum Correlation (Vol Spikes vs Move Direction)
        # ============================================================================
        # Realized volatility at 3 aligned timeframes (1h, 3h, 6h)
        realized_vol_1h = df["abs_return_1h"]  # Current bar volatility
        realized_vol_3h = df["return_1h"].rolling(window=3, min_periods=1).std()
        realized_vol_6h = df["return_1h"].rolling(window=6, min_periods=2).std()

        df["realized_vol_1h"] = realized_vol_1h
        df["realized_vol_3h"] = realized_vol_3h
        df["realized_vol_6h"] = realized_vol_6h

        # Vol ratio changes: recent vs intermediate vs structural
        df["vol_ratio_1h_3h"] = realized_vol_1h / (realized_vol_3h + eps)  # immediate stress
        df["vol_ratio_3h_6h"] = realized_vol_3h / (realized_vol_6h + eps)  # trade duration persistence
        df["vol_ratio_1h_6h"] = realized_vol_1h / (realized_vol_6h + eps)  # overall urgency

        # Volatility spike detection (vol > rolling mean)
        vol_ma_6 = df["abs_return_1h"].rolling(window=6, min_periods=2).mean()
        df["vol_spike_ratio"] = df["abs_return_1h"] / (vol_ma_6 + eps)
        df["is_vol_spike"] = (df["vol_spike_ratio"] > 1.5).astype(float)

        # Momentum magnitude vs volatility (conviction magnitude)
        abs_momentum = df["return_1h"].abs()
        df["momentum_vol_alignment_1h"] = abs_momentum / (realized_vol_1h + eps)
        df["momentum_vol_alignment_3h"] = abs_momentum / (realized_vol_3h + eps)
        df["momentum_vol_alignment_6h"] = abs_momentum / (realized_vol_6h + eps)

        # Volatility-momentum correlation: do vol and momentum move together?
        df["vol_momentum_corr_3h"] = df["abs_return_1h"].rolling(window=3, min_periods=2).apply(
            lambda x: x.corr(df["volume"].iloc[x.index] / (df["volume"].iloc[x.index].mean() + eps))
            if len(x) >= 2 else 0, raw=False
        )
        df["vol_momentum_corr_6h"] = df["abs_return_1h"].rolling(window=6, min_periods=2).apply(
            lambda x: x.corr(df["volume"].iloc[x.index] / (df["volume"].iloc[x.index].mean() + eps))
            if len(x) >= 2 else 0, raw=False
        )

        # Vol-momentum divergence: wicks without body (Ghost signature)
        df["range_momentum_divergence"] = (
            df["range"] - df["abs_return_1h"]
        ) / (df["range"] + eps)  # high = wicks dominate, low = body dominates

        # Vol-momentum synchronization: are spikes aligned with directional moves?
        df["vol_momentum_sync"] = (
            df["is_vol_spike"] * df["volume_direction_conviction"]
        )

        # Smooth vol-momentum metrics
        if config.get("liquidity_use_ewm_features", True):
            df["vol_spike_ratio_ewm3"] = (
                df["vol_spike_ratio"].ewm(span=3, adjust=False).mean()
            )
            df["vol_spike_ratio_ewm6"] = (
                df["vol_spike_ratio"].ewm(span=6, adjust=False).mean()
            )
            df["momentum_vol_alignment_1h_ewm3"] = (
                df["momentum_vol_alignment_1h"].ewm(span=3, adjust=False).mean()
            )
            df["momentum_vol_alignment_3h_ewm3"] = (
                df["momentum_vol_alignment_3h"].ewm(span=3, adjust=False).mean()
            )
            df["range_momentum_divergence_ewm3"] = (
                df["range_momentum_divergence"].ewm(span=3, adjust=False).mean()
            )
            df["range_momentum_divergence_ewm6"] = (
                df["range_momentum_divergence"].ewm(span=6, adjust=False).mean()
            )

        # ============================================================================
        # CATEGORY 4: Orderbook Pressure Proxy (from OHLCV)
        # ============================================================================
        # Estimate orderbook depth from volume distribution over range

        # Volume concentration: orders stacking vs scattered
        close_pct_in_range = df["close_position_range"]
        volume_concentration_3h = (
            (close_pct_in_range * df["volume"]).rolling(window=3, min_periods=1).std() /
            (df["volume"].rolling(window=3, min_periods=1).mean() + eps)
        )
        df["volume_concentration_ratio_3h"] = volume_concentration_3h

        # Order flow imbalance proxy: seller vs buyer pressure
        # >1 = seller pressure, <1 = buyer pressure, ~1 = balanced
        high_move = (df["high"] - df["close"]).abs()
        low_move = (df["close"] - df["low"]).abs()
        df["pressure_ratio"] = (
            (high_move * df["volume"]) / ((low_move * df["volume"]) + eps)
        )

        # Liquidity fill difficulty: volume needed to move price 1%
        price_move_pct = df["abs_return_1h"].clip(lower=0.0001)  # Avoid division by near-zero
        df["kyle_lambda_proxy"] = (
            df["volume"] / price_move_pct
        ).rolling(window=6, min_periods=2).mean()

        # Smooth pressure metrics
        if config.get("liquidity_use_ewm_features", True):
            df["pressure_ratio_ewm3"] = (
                df["pressure_ratio"].ewm(span=3, adjust=False).mean()
            )
            df["pressure_ratio_ewm6"] = (
                df["pressure_ratio"].ewm(span=6, adjust=False).mean()
            )

        # ============================================================================
        # CATEGORY 5: Reversal Patterns (Trap vs Ghost Signature)
        # ============================================================================
        # Reversal intensity: how hard does market reverse?
        sign_changes = (np.sign(df["return_1h"]) != np.sign(df["return_1h"].shift(1))).astype(float)
        df["reversal_intensity"] = df["abs_return_1h"] * sign_changes

        # Post-reversal momentum: do reversals stick?
        df["reversal_conviction"] = (
            (np.sign(df["return_1h"]) == np.sign(df["return_1h"].shift(-1))).astype(float)
            .rolling(window=6, min_periods=2).sum() / 6.0
        )

        # Consecutive reversal bars (whipsaw signature): Ghost = 8-12 per 12 bars, Trend = 2-4
        df["whipsaw_count"] = (
            sign_changes.rolling(window=12, min_periods=4).sum()
        )

        # Reversal-volume alignment: strong vol during reversal?
        df["reversal_volume_sync"] = (
            df["reversal_intensity"] * df["volume_direction_conviction"]
        )

        # Smooth reversal metrics
        if config.get("liquidity_use_ewm_features", True):
            df["reversal_intensity_ewm3"] = (
                df["reversal_intensity"].ewm(span=3, adjust=False).mean()
            )
            df["reversal_conviction_ewm3"] = (
                df["reversal_conviction"].ewm(span=3, adjust=False).mean()
            )
            df["whipsaw_count_ewm6"] = (
                df["whipsaw_count"].ewm(span=6, adjust=False).mean()
            )

        # ============================================================================
        # CATEGORY 6: Multi-Timeframe Volatility Alignment
        # ============================================================================
        # Intra-bar volatility: total range as % of close
        df["intra_bar_vol_estimate"] = (df["high"] - df["low"]) / (df["close"] + eps)

        # Wick-to-body ratio: wicks vs body dominance
        upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
        lower_wick = df[["open", "close"]].min(axis=1) - df["low"]
        df["wick_vol_contribution"] = (upper_wick + lower_wick) / (df["range"] + eps)

        # Session-relative volatility: is this 1h vol high/low for the day (24 bars)?
        df["session_vol_percentile"] = (
            df["abs_return_1h"].rolling(window=24, min_periods=4).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1]
            )
        )

        # Vol clustering: does vol persist or scatter?
        vol_above_ma = (df["abs_return_1h"] > df["abs_return_1h"].rolling(window=6).mean()).astype(float)
        df["vol_clustering"] = vol_above_ma.rolling(window=6, min_periods=2).sum() / 6.0

        # Vol regime change detection: spiking or dying?
        df["vol_regime_change"] = (
            (realized_vol_3h - realized_vol_6h) / (realized_vol_6h + eps)
        )

        # Smooth multi-TF metrics
        if config.get("liquidity_use_ewm_features", True):
            df["session_vol_percentile_ewm6"] = (
                df["session_vol_percentile"].ewm(span=6, adjust=False).mean()
            )
            df["vol_clustering_ewm6"] = (
                df["vol_clustering"].ewm(span=6, adjust=False).mean()
            )
            df["vol_regime_change_ewm6"] = (
                df["vol_regime_change"].ewm(span=6, adjust=False).mean()
            )

        # ============================================================================
        # CATEGORY 7: Information Efficiency Metrics (Price Discovery Quality)
        # ============================================================================
        # Efficiency ratio (Kaufman): directional move vs total volatility
        price_change_6h = (df["close"] - df["close"].shift(6)).abs()
        volatility_6h = df["abs_return_1h"].rolling(window=6, min_periods=2).sum()
        df["efficiency_ratio"] = (
            price_change_6h / (volatility_6h + eps)
        )

        # Return predictability: can today's move predict tomorrow's?
        df["return_autocorr_lag6"] = df["return_1h"].rolling(window=12, min_periods=6).apply(
            lambda x: x.iloc[:6].corr(x.iloc[6:]) if len(x) == 12 else 0, raw=False
        )

        # Volume-price trend synchronization: do volume and price move together?
        price_trend_6h = (df["close"].diff(6) > 0).astype(float)
        volume_trend_6h = (df["volume"] > df["volume"].rolling(window=6).mean()).astype(float)
        df["volume_price_trend_sync"] = (
            price_trend_6h.rolling(window=6, min_periods=2).mean() -
            volume_trend_6h.rolling(window=6, min_periods=2).mean()
        )

        # Market microstructure quality: price impact per unit volume
        df["price_impact_ratio"] = (
            df["range"] / (df["volume"] + eps)
        )

        # Smooth efficiency metrics
        if config.get("liquidity_use_ewm_features", True):
            df["efficiency_ratio_ewm6"] = (
                df["efficiency_ratio"].ewm(span=6, adjust=False).mean()
            )
            df["return_autocorr_lag6_ewm6"] = (
                df["return_autocorr_lag6"].ewm(span=6, adjust=False).mean()
            )
            df["price_impact_ratio_ewm6"] = (
                df["price_impact_ratio"].ewm(span=6, adjust=False).mean()
            )

        return df

    def _compute_kde_threshold(self, series: pd.Series, config: Dict[str, Any], prefix: str) -> float:
        vals = series.dropna().astype(float)
        if len(vals) < 50:
            return float(vals.median()) if len(vals) > 0 else 0.0

        q33 = vals.quantile(0.33)
        q66 = vals.quantile(0.66)
        band_vals = vals[(vals >= q33) & (vals <= q66)]
        if len(band_vals) < 20:
            return float(vals.median())

        try:
            kde = gaussian_kde(band_vals.values.astype(float))
            grid = np.linspace(q33, q66, 256)
            densities = kde(grid)
            idx_min = int(np.argmin(densities))
            thresh = float(grid[idx_min])
            tprint_info(
                f"KDE threshold for {prefix}: q33={q33:.4f}, q66={q66:.4f}, thresh={thresh:.4f}"
            )
            return thresh
        except Exception as exc:
            tprint_warning(f"KDE threshold estimation failed for {prefix}: {exc}; using median")
            return float(vals.median())

    def _assign_liquidity_regimes(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        work_df = df.copy()

        if "rvol_24" not in work_df.columns or "normalized_range" not in work_df.columns:
            raise ValueError("Missing rvol_24 or normalized_range for liquidity regime assignment")

        vol_thresh = self._compute_kde_threshold(work_df["rvol_24"], config, prefix="volume_rvol_24")
        range_thresh = self._compute_kde_threshold(work_df["normalized_range"], config, prefix="normalized_range")

        work_df["volume_state"] = np.where(work_df["rvol_24"] >= vol_thresh, 1, 0)
        work_df["move_state"] = np.where(work_df["normalized_range"] >= range_thresh, 1, 0)

        regimes = np.full(len(work_df), np.nan, dtype=float)

        high_vol = work_df["volume_state"] == 1
        high_move = work_df["move_state"] == 1
        low_move = work_df["move_state"] == 0

        # 1 = Valid Trend (high vol, high move)
        mask_valid = high_vol & high_move

        # 2 = Absorption (high vol, low move or absorption_ratio > 1)
        mask_absorption = high_vol & low_move
        if "absorption_ratio" in work_df.columns:
            absorption_ratio_thresh = float(config.get("liquidity_absorption_ratio_thresh", 1.5))
            mask_absorption = mask_absorption & (work_df["absorption_ratio"] > absorption_ratio_thresh)

        # 3 = Ghost / Drift (low vol, high move or ghost_ratio > 1)
        mask_ghost = (work_df["volume_state"] == 0) & (work_df["move_state"] == 1)
        if "ghost_ratio" in work_df.columns:
            ghost_ratio_thresh = float(config.get("liquidity_ghost_ratio_thresh", 1.5))
            mask_ghost |= work_df["ghost_ratio"] > ghost_ratio_thresh

        # 0 = Apathy (everything else)
        mask_apathy = ~(mask_valid | mask_absorption | mask_ghost)

        regimes[mask_apathy.values] = 0
        regimes[mask_absorption.values] = 2
        regimes[mask_ghost.values] = 3
        regimes[mask_valid.values] = 1

        work_df["liquidity_regime"] = regimes

        # Ambiguity: enforce nearest regime but encode low confidence via weights
        d_vol = (work_df["rvol_24"] - vol_thresh).abs()
        d_range = (work_df["normalized_range"] - range_thresh).abs()

        # Use local scale from central band for normalization
        vol_vals = work_df["rvol_24"].dropna()
        range_vals = work_df["normalized_range"].dropna()
        vol_band_width = max(vol_vals.quantile(0.66) - vol_vals.quantile(0.33), 1e-6) if len(vol_vals) > 20 else 1.0
        range_band_width = max(range_vals.quantile(0.66) - range_vals.quantile(0.33), 1e-6) if len(range_vals) > 20 else 1.0

        d_vol_norm = (d_vol / vol_band_width).clip(0.0, 1.0)
        d_range_norm = (d_range / range_band_width).clip(0.0, 1.0)
        ambiguity = 1.0 - np.minimum(d_vol_norm, d_range_norm)  # 1.0 near threshold, 0 far away

        w_min = float(config.get("liquidity_min_sample_weight", 0.3))
        sample_weight = w_min + (1.0 - w_min) * (1.0 - ambiguity)

        work_df["liquidity_sample_weight"] = sample_weight.astype(float)

        return work_df

    def _train_liquidity_model(
        self,
        liquidity_df: pd.DataFrame,
        config: Dict[str, Any],
    ) -> Tuple[Any, Optional[pd.DataFrame], pd.Series, Dict[str, Any], Dict[str, Any]]:
        """Train an XGBClassifier to predict liquidity regimes (0–3)."""
        try:
            from xgboost import XGBClassifier
        except ImportError as e:
            raise ImportError("xgboost is required for liquidity regime model training") from e

        try:
            from sklearn.metrics import (
                accuracy_score,
                f1_score,
                confusion_matrix,
            )
            from sklearn.calibration import CalibratedClassifierCV
        except ImportError:
            accuracy_score = None  # type: ignore[assignment]
            f1_score = None  # type: ignore[assignment]
            confusion_matrix = None  # type: ignore[assignment]
            CalibratedClassifierCV = None  # type: ignore[assignment]

        df = liquidity_df.copy()
        if "liquidity_regime" not in df.columns:
            raise ValueError("liquidity_regime column not found in dataset")

        df = df.dropna(subset=["liquidity_regime"])
        if df.empty:
            raise ValueError("No valid samples for liquidity model training after dropping NaNs")

        y = df["liquidity_regime"].astype(int)

        # Map observed labels to contiguous indices for XGBoost compatibility
        unique_labels = sorted(y.unique())
        if not unique_labels:
            raise ValueError("No unique liquidity_regime labels available for model training")

        label_to_new: Dict[int, int] = {int(lbl): idx for idx, lbl in enumerate(unique_labels)}
        new_to_label: Dict[int, int] = {idx: int(lbl) for lbl, idx in label_to_new.items()}

        numeric_df = df.select_dtypes(include=[np.number])
        drop_cols = ["liquidity_regime"]
        feature_cols = [c for c in numeric_df.columns if c not in drop_cols]
        if not feature_cols:
            raise ValueError("No numeric features available for liquidity model training")

        X = numeric_df[feature_cols]

        min_samples = int(config.get("liquidity_min_samples", 200))
        if len(X) < max(min_samples, 50):
            raise ValueError(
                f"Insufficient samples for liquidity model training: {len(X)} < {min_samples}"
            )

        train_frac = float(config.get("liquidity_train_fraction", 0.8))
        train_frac = min(max(train_frac, 0.5), 0.95)
        split_idx = int(len(X) * train_frac)
        split_idx = max(min(split_idx, len(X) - 1), 1)

        X_train_raw, y_train = X.iloc[:split_idx].copy(), y.iloc[:split_idx]
        X_val_raw, y_val = X.iloc[split_idx:].copy(), y.iloc[split_idx:]

        # Use mapped labels for training/calibration
        y_train_mapped = y_train.map(label_to_new).astype(int)
        y_val_mapped = y_val.map(label_to_new).astype(int)

        # Scaling
        normalizer_config: Dict[str, Any] = {
            "default_strategy": "robust",
            "auto_select": False,
            "handle_outliers": True,
            "outlier_threshold": float(config.get("liquidity_outlier_threshold", 3.0)),
            "use_vectorbt": False,
        }
        scaler = ScalingNormalizer(normalizer_config)
        X_train_scaled = scaler.fit_transform(X_train_raw, strategy="robust")
        X_val_scaled = scaler.transform(X_val_raw)
        X_scaled_full = scaler.transform(X)

        # Optional EWM smoothing
        use_ewm_features = bool(config.get("liquidity_use_ewm_features", True))
        ewma_periods_cfg = config.get("liquidity_ewm_periods", [2, 6, 10])
        try:
            ewma_periods = [int(p) for p in ewma_periods_cfg if int(p) > 0]
        except Exception:
            ewma_periods = [2, 6, 10]

        if use_ewm_features and ewma_periods:
            base_df = X_scaled_full.copy()
            feature_names_seq: List[str] = list(base_df.columns)
            aggregated_ewm: Optional[np.ndarray] = None
            n_features = base_df.shape[1]

            for period in ewma_periods:
                alpha_val = 2.0 / float(period + 1)
                try:
                    smoothed_array, _ = apply_ewm_smoothing(
                        base_df.values,
                        alpha=alpha_val,
                        feature_names=feature_names_seq,
                        use_vectorization_optimization=False,
                    )
                    if smoothed_array.shape[1] < 2 * n_features:
                        raise ValueError(
                            f"Unexpected smoothed_array shape {smoothed_array.shape} for n_features={n_features}"
                        )
                    ewm_block = smoothed_array[:, n_features:]
                    if aggregated_ewm is None:
                        aggregated_ewm = ewm_block.astype(float)
                    else:
                        aggregated_ewm = aggregated_ewm + ewm_block.astype(float)
                except Exception as e:
                    tprint_warning(
                        f"EWMA temporal smoothing failed for period={period} (using unsmoothed features): {e}"
                    )
                    aggregated_ewm = None
                    break

            if aggregated_ewm is not None:
                aggregated_ewm = aggregated_ewm / float(len(ewma_periods))
                features_df = pd.DataFrame(
                    aggregated_ewm,
                    index=base_df.index,
                    columns=pd.Index(feature_names_seq),
                )
                X_features_full = features_df
                X_train = X_features_full.iloc[:split_idx].copy()
                X_val = X_features_full.iloc[split_idx:].copy()
                X_scaled_full = X_features_full
                extended_feature_names = feature_names_seq
            else:
                X_train = X_train_scaled
                X_val = X_val_scaled
                extended_feature_names = list(X_scaled_full.columns)
        else:
            X_train = X_train_scaled
            X_val = X_val_scaled
            extended_feature_names = list(X_scaled_full.columns)

        # Use sample weights from ambiguity handling if available
        sample_weight = liquidity_df.get("liquidity_sample_weight")
        sw_train = None
        if sample_weight is not None:
            # Align weights with filtered df index and enforce strict positivity
            sw = sample_weight.loc[df.index].astype(float)
            sw = sw.replace([np.inf, -np.inf], np.nan).fillna(1.0)
            sw = sw.clip(lower=1e-6)
            sw_train = sw.iloc[:split_idx].values

        training_metrics: Dict[str, Any] = {}
        training_metrics["model_type"] = "xgboost_multiclass"

        feature_pipeline_artifacts: Dict[str, Any] = {
            "feature_names": extended_feature_names,
            "scaler": scaler,
            "normalizer_config": normalizer_config,
        }

        base_params: Dict[str, Any] = {
            "objective": "multi:softprob",
            "num_class": len(unique_labels),
            "n_estimators": int(config.get("liquidity_n_estimators", 300)),
            "learning_rate": float(config.get("liquidity_learning_rate", 0.05)),
            "max_depth": int(config.get("liquidity_max_depth", 5)),
            "subsample": float(config.get("liquidity_subsample", 0.8)),
            "colsample_bytree": float(config.get("liquidity_colsample_bytree", 0.8)),
            "random_state": int(config.get("liquidity_random_state", 42)),
            "n_jobs": int(config.get("liquidity_n_jobs", -1)),
        }

        model = XGBClassifier(**base_params)
        model.fit(X_train, y_train_mapped, sample_weight=sw_train)

        # Evaluate on validation set
        proba_val = model.predict_proba(X_val) if len(X_val) > 0 else None
        if proba_val is not None and accuracy_score is not None and f1_score is not None:
            y_val_pred = np.argmax(proba_val, axis=1)
            training_metrics["val_accuracy_uncalibrated"] = float(accuracy_score(y_val_mapped, y_val_pred))
            training_metrics["val_f1_macro_uncalibrated"] = float(
                f1_score(y_val_mapped, y_val_pred, average="macro")
            )

        # Probability calibration (multi-class via one-vs-rest)
        calibration_enabled = bool(config.get("liquidity_enable_prob_calibration", True))
        training_metrics["probability_calibration_enabled"] = calibration_enabled

        calibrated_model = model
        if (
            calibration_enabled
            and CalibratedClassifierCV is not None
            and len(X_val) > 0
        ):
            try:
                calibrated_model = CalibratedClassifierCV(
                    base_estimator=model,
                    method="isotonic",
                    cv="prefit",
                )
                calibrated_model.fit(X_val, y_val_mapped)
                training_metrics["calibration_method"] = "isotonic_regression"
            except Exception as calib_err:
                tprint_warning(f"Liquidity probability calibration failed: {calib_err}")
                calibrated_model = model

        # Probabilities on full dataset (model index space)
        proba_all = calibrated_model.predict_proba(X_scaled_full)

        # Map model probabilities back to canonical regime ids
        proba_df = pd.DataFrame(index=df.index)
        for old_label, new_idx in label_to_new.items():
            proba_df[f"p_regime_{old_label}"] = proba_all[:, new_idx]

        # Ensure columns exist for all canonical regimes up to configured n_regimes
        n_regimes_cfg = int(config.get("liquidity_n_regimes", 4))
        for lbl in range(n_regimes_cfg):
            col_name = f"p_regime_{lbl}"
            if col_name not in proba_df.columns:
                proba_df[col_name] = 0.0

        # Core ghost probability feature (Regime 3 if present)
        if 3 in label_to_new and "p_regime_3" in proba_df.columns:
            proba_df["liquidity_regime_prob_ghost"] = proba_df["p_regime_3"]

        return calibrated_model, proba_df, y, training_metrics, feature_pipeline_artifacts

    def _assess_liquidity_regime_quality(
        self,
        *,
        liquidity_df: pd.DataFrame,
        regime_col: Optional[str],
        config: Dict[str, Any],
    ) -> Tuple[Optional[ClusterQualityMetrics], Optional[str]]:
        if regime_col is None or regime_col not in liquidity_df.columns:
            tprint_warning("No liquidity regime column provided; skipping regime quality assessment")
            return None, None

        regime_series = liquidity_df[regime_col]
        valid_mask = regime_series.notna()
        if valid_mask.sum() == 0:
            tprint_warning("No valid liquidity regime labels for quality assessment")
            return None, None

        regime_labels = np.asarray(regime_series[valid_mask].astype(int), dtype=int)

        numeric_df = liquidity_df.select_dtypes(include=[np.number])
        feature_cols = [c for c in numeric_df.columns if c != regime_col]
        if not feature_cols:
            tprint_warning("No numeric features available for liquidity regime quality assessment")
            return None, None

        feature_data = numeric_df[feature_cols].loc[valid_mask]
        timestamps = liquidity_df.index[valid_mask]

        min_regime_size = int(config.get("liquidity_min_regime_size", 3))
        temporal_mode = str(config.get("liquidity_temporal_sensitivity_mode", "regime_persistence_focused"))
        fast_mode = bool(config.get("liquidity_quality_fast_mode", False))

        try:
            metrics = self.quality_assessor.assess_quality(
                regime_labels=regime_labels,
                feature_data=feature_data,
                forward_returns=None,
                timestamps=timestamps,
                min_regime_size=min_regime_size,
                temporal_sensitivity_mode=temporal_mode,
                fast_mode=fast_mode,
                standardize_for_metrics=True,
            )
        except Exception as exc:
            tprint_warning(f"Liquidity regime quality assessment failed: {exc}")
            return None, None

        metrics_dict: Dict[str, Any]
        if hasattr(metrics, "to_dict"):
            metrics_dict = metrics.to_dict()  # type: ignore[assignment]
        elif is_dataclass(metrics) and not isinstance(metrics, type):
            metrics_dict = asdict(metrics)
        else:
            metrics_dict = {"metrics": metrics}

        quality_df = pd.DataFrame([metrics_dict])
        try:
            quality_path = self._save_artifact(
                data=quality_df,
                artifact_name="ml_liquidity_regime_quality_1h",
                artifact_type="data",
                metadata={
                    "min_regime_size": min_regime_size,
                },
            )
        except Exception as save_exc:
            tprint_warning(f"Failed to save liquidity regime quality artifact: {save_exc}")
            quality_path = None

        return metrics, quality_path

    def _map_probabilities_to_15m(
        self,
        *,
        proba_df: Optional[pd.DataFrame],
        market_data_1h: pd.DataFrame,
        symbol: str,
        exchange: str,
        direction: str,
        config: Dict[str, Any],
    ) -> Optional[pd.DataFrame]:
        if proba_df is None or proba_df.empty:
            return None

        output_timeframe = str(config.get("liquidity_output_timeframe", "15m"))
        if output_timeframe == "1h":
            return proba_df

        # Load 15m market data to get target index
        market_data_15m, _ = self.load_market_data_or_fail(
            {**config, "timeframe": output_timeframe},
            pipeline_state={},
            allow_config_override=True,
        )

        if not isinstance(market_data_15m, pd.DataFrame) or market_data_15m.empty:
            tprint_warning("15m market data unavailable; skipping 1h→15m probability mapping")
            return None

        if not isinstance(market_data_15m.index, pd.DatetimeIndex):
            market_data_15m = market_data_15m.copy()
            market_data_15m.index = pd.to_datetime(market_data_15m.index)

        # Ensure 1h index is DatetimeIndex
        if not isinstance(proba_df.index, pd.DatetimeIndex):
            idx = pd.to_datetime(proba_df.index)
            proba_df = proba_df.copy()
            proba_df.index = idx

        mode = str(config.get("liquidity_prob_interpolation_mode", "step")).lower()

        if mode == "linear":
            # Linear interpolation between consecutive 1h bars across 15m children
            one_h_index = proba_df.index.sort_values()
            if len(one_h_index) < 2:
                return proba_df.reindex(market_data_15m.index, method="ffill")

            # Build a continuous 15m index spanning the 1h data
            full_15m_index = pd.date_range(
                start=one_h_index.min(),
                end=one_h_index.max(),
                freq=output_timeframe,
            )
            # Reindex to 15m and interpolate linearly in time
            step_reindexed = proba_df.reindex(one_h_index)
            step_resampled = step_reindexed.resample(output_timeframe).ffill()
            interp_df = step_resampled.reindex(full_15m_index).interpolate(method="time")
            # Align to actual 15m market index
            mapped = interp_df.reindex(market_data_15m.index, method="nearest")
            return mapped

        # Default: step mapping from parent 1h bar via floor
        parent_index = market_data_15m.index.floor("1H")
        mapped = proba_df.reindex(parent_index, method="ffill")
        mapped.index = market_data_15m.index
        return mapped
