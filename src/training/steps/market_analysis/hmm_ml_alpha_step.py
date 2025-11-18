"""
HMM ML Alpha Step

This step consumes 1h Rolling HMM regime outputs plus OHLCV data to
construct forward-return-based alpha labels and a cleaned training
DataFrame for downstream models (e.g., regime-aware 15m models).

Responsibilities (initial version):
- Load 1h HMM artifacts from versioned HDF5 (labels, probabilities,
  economic features) using the same context as
  RollingHMMRegimeDiscoveryStep.
- Load 1h OHLCV market data.
- Align all series on a common DatetimeIndex.
- Compute forward 1h log returns and a simple binary alpha target.
- Save the resulting bar-level dataset into a dedicated
  versioned_artifacts store for later consumption.

Model training and regime-level alpha statistics will be added in
follow-up iterations.
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
)
from src.features_common.transforms.scaling_normalization import ScalingNormalizer
from src.training.steps.market_analysis.clusters.cluster_quality_assessor import (
    ClusterQualityAssessor,
    ClusterQualityMetrics,
)
from src.utils.ml_common.optimization import (
    HierarchicalParameterOptimizer,
    ParameterGroup,
    OptimizationStage,
)
from src.utils.ml_common.feature_engineering.feature_smoothing import apply_ewm_smoothing


logger = logging.getLogger(__name__)


class HMMMLAlphaStep(BaseStep):
    """Pipeline step to construct alpha labels from 1h Rolling HMM regimes."""

    def __init__(self, step_name: str = "hmm_ml_alpha_step"):
        """Initialize the HMM ML alpha step with versioned artifacts enabled."""
        super().__init__(step_name, use_versioned_artifacts=True)
        self.logger = logger.getChild("HMMMLAlphaStep") if hasattr(logger, "getChild") else logger
        tprint(f"✅ Initialized {step_name} step", "SUCCESS")

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the alpha label construction from 1h HMM regimes.

        Expected config keys (minimum):
            - symbol: Trading symbol (e.g., 'ETHUSDT')
            - exchange: Exchange name (e.g., 'binance')
            - regime_timeframe: Timeframe used for HMM regimes (default: '1h')
            - direction: Trading direction (default: 'long')
            - execution_mode: 'full', 'light', 'blank', etc. (used for data loading)

        Optional alpha configuration:
            - alpha_horizon_bars: Forward horizon in bars (default: 1)
            - alpha_return_type: 'log' or 'simple' (default: 'log')
            - alpha_target_type: 'classification' or 'regression'
              (default: 'classification')
        """
        start_time = time.time()

        try:
            symbol = str(config.get("symbol", "ETHUSDT"))
            exchange = str(config.get("exchange", "binance"))
            regime_timeframe = str(
                config.get("regime_timeframe", config.get("timeframe", "1h"))
            )
            direction = str(config.get("direction", "long"))

            # Set default alpha-specific configuration if not provided
            # Smoothing defaults are chosen to target ~3–5 bar regime persistence
            # while allowing overrides via config/CLI when needed.
            alpha_defaults: Dict[str, Any] = {
                "alpha_target_smoothing_method": "ewm",
                "alpha_target_smoothing_window": 3,
                "alpha_score_smoothing_method": "ewm",
                "alpha_score_smoothing_window": 6,
                # Keep HPO opt-in; these defaults are used when explicitly enabled
                "alpha_enable_hpo": False,
                "alpha_hpo_cv_folds": 3,
                "alpha_hpo_final_trials": 20,
            }
            for k, v in alpha_defaults.items():
                config.setdefault(k, v)

            if not symbol or not exchange:
                raise ValueError("Config must include 'symbol' and 'exchange'")

            tprint_info(
                f"🚀 Starting {self.step_name} for {symbol} on {exchange} "
                f"(regime_timeframe={regime_timeframe})"
            )

            # ------------------------------------------------------------------
            # 1) Load HMM artifacts (labels/probabilities/economic features)
            # ------------------------------------------------------------------
            # Use the same context as RollingHMMRegimeDiscoveryStep when loading
            # its artifacts (model='regime').
            self.set_context(
                symbol=symbol,
                exchange=exchange,
                timeframe=regime_timeframe,
                direction=direction,
                model="regime",
            )

            labels_df, probs_df, economic_df = self._load_hmm_artifacts(
                symbol=symbol,
                exchange=exchange,
                timeframe=regime_timeframe,
                config=config,
            )

            # ------------------------------------------------------------------
            # 2) Load 1h OHLCV market data
            # ------------------------------------------------------------------
            market_data, market_source = self.load_market_data_or_fail(
                {
                    **config,
                    "timeframe": regime_timeframe,
                },
                pipeline_state={},
                allow_config_override=True,
            )

            if not isinstance(market_data, pd.DataFrame) or market_data.empty:
                raise ValueError("Loaded market data is empty or not a DataFrame")

            if not isinstance(market_data.index, pd.DatetimeIndex):
                try:
                    market_data = market_data.copy()
                    # Handle tz-aware objects by converting to UTC then dropping tz
                    try:
                        market_data.index = pd.to_datetime(market_data.index)
                    except (TypeError, ValueError):
                        market_data.index = pd.to_datetime(market_data.index, utc=True)
                        market_data.index = market_data.index.tz_convert(None)
                except Exception as exc:  # pragma: no cover - defensive
                    raise ValueError(
                        "Market data index could not be converted to DatetimeIndex"
                    ) from exc

            tprint_info(
                f"✅ Loaded market data from {market_source}: {market_data.shape} "
                f"({market_data.index.min()} → {market_data.index.max()})"
            )

            # ------------------------------------------------------------------
            # 3) Align all inputs on common DatetimeIndex
            # ------------------------------------------------------------------
            aligned_df = self._align_inputs(
                market_data=market_data,
                labels_df=labels_df,
                probs_df=probs_df,
                economic_df=economic_df,
            )

            if aligned_df.empty:
                raise ValueError("Aligned dataset is empty after merging inputs")

            # ------------------------------------------------------------------
            # 4) Compute alpha labels
            # ------------------------------------------------------------------
            alpha_df = self._compute_alpha_labels(aligned_df, config)

            if alpha_df.empty:
                raise ValueError("Alpha dataset is empty after label construction")

            # ------------------------------------------------------------------
            # 5) Train LightGBM alpha model and derive alpha regimes
            # ------------------------------------------------------------------
            model = None
            alpha_scores = None
            training_metrics: Dict[str, Any] = {}
            regime_stats_df: Optional[pd.DataFrame] = None
            model_path: Optional[str] = None
            regime_stats_path: Optional[str] = None
            regime_col_name: Optional[str] = None
            alpha_quality_metrics: Optional[ClusterQualityMetrics] = None
            alpha_quality_path: Optional[str] = None
            feature_pipeline_artifacts: Optional[Dict[str, Any]] = None
            feature_pipeline_path: Optional[str] = None

            try:
                model, alpha_scores, pred_col_name, training_metrics, feature_pipeline_artifacts = self._train_alpha_model(
                    alpha_df,
                    config,
                )
                if alpha_scores is not None:
                    alpha_df[pred_col_name] = alpha_scores
                    alpha_df, regime_stats_df, regime_col_name = self._assign_alpha_regimes(
                        alpha_df,
                        alpha_scores,
                        config,
                    )
            except ImportError as lgb_err:
                tprint_warning(
                    f"LightGBM not available; skipping alpha model training: {lgb_err}"
                )
            except Exception as model_exc:
                tprint_warning(
                    f"Alpha model training failed; continuing with labels only: {model_exc}"
                )

            if alpha_scores is None or regime_col_name is None:
                try:
                    forward_ret_cols = [
                        col
                        for col in alpha_df.columns
                        if col.startswith("alpha_forward_return_")
                    ]
                    if forward_ret_cols:
                        fallback_series = alpha_df[forward_ret_cols[0]].astype(float)
                        valid_count = fallback_series.notna().sum()
                        if valid_count >= 3:
                            pred_col_name_fallback = f"alpha_fallback_score_{forward_ret_cols[0].split('_')[-1]}"
                            alpha_scores = fallback_series
                            alpha_df[pred_col_name_fallback] = alpha_scores
                            alpha_df, regime_stats_df, regime_col_name = self._assign_alpha_regimes(
                                alpha_df,
                                alpha_scores,
                                config,
                            )
                            training_metrics["alpha_fallback_used"] = True
                            training_metrics["alpha_fallback_source"] = "forward_return"
                        else:
                            tprint_warning(
                                f"Not enough samples ({valid_count}) for fallback alpha regime assignment"
                            )
                except Exception as fallback_exc:
                    tprint_warning(
                        f"Alpha fallback regime assignment failed; proceeding without regimes: {fallback_exc}"
                    )

            # ------------------------------------------------------------------
            # 6) Switch context to dedicated alpha namespace and assess quality
            # ------------------------------------------------------------------
            # Switch context to a dedicated alpha model namespace so we do not
            # pollute the original HMM regime store.
            self.set_context(
                symbol=symbol,
                exchange=exchange,
                timeframe=regime_timeframe,
                direction=direction,
                model="regime_alpha",
            )

            # Run unified cluster quality assessment on alpha regimes (if any)
            try:
                alpha_quality_metrics, alpha_quality_path = self._assess_alpha_regime_quality(
                    alpha_df=alpha_df,
                    regime_col=regime_col_name,
                    config=config,
                )
            except Exception as quality_exc:
                tprint_warning(f"Alpha regime quality assessment failed: {quality_exc}")

            alpha_to_save = alpha_df.reset_index().rename(columns={alpha_df.index.name or "index": "timestamp"})

            tprint_info(
                f"💾 Saving alpha training dataset with shape {alpha_to_save.shape} "
                f"to versioned HDF5 store"
            )
            training_data_path = self._save_artifact(
                data=alpha_to_save,
                artifact_name="hmm_alpha_training_data_1h",
                artifact_type="data",
                metadata={
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": regime_timeframe,
                    "source_market_data": market_source,
                },
            )

            # Save trained model if available
            if model is not None:
                try:
                    tprint_info("💾 Saving LightGBM alpha model via artifact router")
                    model_path = self._save_artifact(
                        data=model,
                        artifact_name="hmm_alpha_model_1h",
                        artifact_type="model",
                        metadata={
                            "symbol": symbol,
                            "exchange": exchange,
                            "timeframe": regime_timeframe,
                            "model_type": "lightgbm",
                        },
                    )
                except Exception as save_model_exc:
                    tprint_warning(f"Failed to save alpha model artifact: {save_model_exc}")

            # Persist feature pipeline (feature list + scaler state) for live usage
            if feature_pipeline_artifacts is not None:
                try:
                    feature_pipeline_path = self._save_artifact(
                        data=feature_pipeline_artifacts,
                        artifact_name="hmm_alpha_feature_pipeline_1h",
                        artifact_type="model",
                        metadata={
                            "symbol": symbol,
                            "exchange": exchange,
                            "timeframe": regime_timeframe,
                            "feature_names": feature_pipeline_artifacts.get("feature_names", []),
                        },
                    )
                except Exception as save_fp_exc:
                    tprint_warning(f"Failed to save alpha feature pipeline artifact: {save_fp_exc}")

            # Save regime-level alpha statistics if available
            if regime_stats_df is not None and not regime_stats_df.empty:
                try:
                    regime_stats_to_save = regime_stats_df.reset_index()
                    regime_stats_path = self._save_artifact(
                        data=regime_stats_to_save,
                        artifact_name="hmm_alpha_regime_stats_1h",
                        artifact_type="data",
                        metadata={
                            "symbol": symbol,
                            "exchange": exchange,
                            "timeframe": regime_timeframe,
                        },
                    )
                except Exception as save_stats_exc:
                    tprint_warning(
                        f"Failed to save alpha regime statistics artifact: {save_stats_exc}"
                    )

            execution_time = time.time() - start_time
            tprint_info(
                f"✅ {self.step_name} completed in {execution_time:.2f}s "
                f"with {len(alpha_df)} samples"
            )

            # ------------------------------------------------------------------
            # 7) Generate alpha-specific reports in outcomes/
            # ------------------------------------------------------------------
            try:
                # Alpha quality markdown + CSV reports via ClusterQualityAssessor
                if alpha_quality_metrics is not None:
                    try:
                        method_config = {
                            "alpha_config": {
                                "alpha_horizon_bars": config.get("alpha_horizon_bars", 1),
                                "alpha_regime_bins": config.get("alpha_regime_bins", 5),
                                "alpha_target_type": config.get("alpha_target_type", "regression"),
                            }
                        }

                        report_prefix = "hmm_alpha_quality"
                        self.quality_assessor.generate_markdown_report(
                            alpha_quality_metrics,
                            symbol=symbol,
                            output_dir="outcomes",
                            method_specific_config=method_config,
                            report_prefix=report_prefix,
                        )

                        self.quality_assessor.generate_comprehensive_csv_report(
                            alpha_quality_metrics,
                            all_trials=None,
                            symbol=symbol,
                            output_dir="outcomes",
                            method_specific_config=method_config,
                        )
                    except Exception as report_exc:
                        tprint_warning(f"Alpha quality report generation failed (ignored): {report_exc}")

                # Regime stats CSV for alpha regimes
                if regime_stats_df is not None and not regime_stats_df.empty:
                    try:
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        stats_csv_name = f"hmm_alpha_regime_stats_{symbol}_{ts}.csv"
                        stats_csv_path = f"outcomes/{stats_csv_name}"
                        regime_stats_df.to_csv(stats_csv_path)
                        tprint_info(
                            f"📊 Saved alpha regime statistics CSV: {stats_csv_path}"
                        )
                    except Exception as stats_csv_exc:
                        tprint_warning(
                            f"Failed to save alpha regime statistics CSV (ignored): {stats_csv_exc}"
                        )
            except Exception as report_outer_exc:
                tprint_warning(
                    f"Alpha report generation encountered a non-fatal error: {report_outer_exc}"
                )

            return {
                "success": True,
                "artifacts": {
                    "alpha_training_data": alpha_df,
                    "alpha_training_data_path": training_data_path,
                    "alpha_model_path": model_path,
                    "alpha_regime_stats": regime_stats_df,
                    "alpha_regime_stats_path": regime_stats_path,
                    "alpha_regime_quality_metrics": alpha_quality_metrics,
                    "alpha_regime_quality_path": alpha_quality_path,
                    "alpha_feature_pipeline": feature_pipeline_artifacts,
                    "alpha_feature_pipeline_path": feature_pipeline_path,
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
    # Internal helper methods
    # ------------------------------------------------------------------

    def _load_hmm_artifacts(
        self,
        *,
        symbol: str,
        exchange: str,
        timeframe: str,
        config: Dict[str, Any],
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Load HMM labels, probabilities, and economic features from HDF5.

        Uses the same context that RollingHMMRegimeDiscoveryStep used when
        saving its artifacts: model='regime' and the specified timeframe.
        """
        # Labels
        labels = self._get_artifact(
            artifact_name="rolling_hmm_regime_labels",
            artifact_type="data",
        )
        if labels is None:
            raise FileNotFoundError(
                "rolling_hmm_regime_labels artifact not found in versioned store"
            )

        if not isinstance(labels, pd.DataFrame):
            labels = pd.DataFrame(labels)

        if "timestamp" in labels.columns:
            labels = labels.copy()
            labels["timestamp"] = pd.to_datetime(labels["timestamp"])
            labels.set_index("timestamp", inplace=True)
        elif isinstance(labels.index, pd.DatetimeIndex):
            pass
        else:
            raise ValueError(
                "HMM labels artifact must have a 'timestamp' column or DatetimeIndex"
            )

        tprint_info(
            f"✅ Loaded HMM labels: {labels.shape} "
            f"({labels.index.min()} → {labels.index.max()})"
        )

        # Probabilities (optional)
        probs = self._get_artifact(
            artifact_name="rolling_hmm_regime_probabilities",
            artifact_type="data",
        )
        if probs is not None:
            if not isinstance(probs, pd.DataFrame):
                probs = pd.DataFrame(probs)
            if "timestamp" in probs.columns:
                probs = probs.copy()
                probs["timestamp"] = pd.to_datetime(probs["timestamp"])
                probs.set_index("timestamp", inplace=True)
            elif not isinstance(probs.index, pd.DatetimeIndex):
                tprint_warning(
                    "HMM probabilities artifact has no DatetimeIndex; "
                    "dropping probability features"
                )
                probs = None

        if probs is not None and not probs.empty:
            tprint_info(
                f"✅ Loaded HMM probabilities: {probs.shape} "
                f"({probs.index.min()} → {probs.index.max()})"
            )
        else:
            tprint_warning("No HMM probabilities found; proceeding without them")
            probs = None

        # Economic features (optional)
        economic = self._get_artifact(
            artifact_name="rolling_hmm_economic_features",
            artifact_type="data",
        )
        if economic is not None:
            if not isinstance(economic, pd.DataFrame):
                economic = pd.DataFrame(economic)
            if "timestamp" in economic.columns:
                economic = economic.copy()
                economic["timestamp"] = pd.to_datetime(economic["timestamp"])
                economic.set_index("timestamp", inplace=True)
            elif not isinstance(economic.index, pd.DatetimeIndex):
                tprint_warning(
                    "Economic features artifact has no DatetimeIndex; "
                    "dropping economic features"
                )
                economic = None

        if economic is not None and not economic.empty:
            tprint_info(
                f"✅ Loaded economic features: {economic.shape} "
                f"({economic.index.min()} → {economic.index.max()})"
            )
        else:
            tprint_warning("No economic features found; proceeding without them")
            economic = None

        return labels, probs, economic

    def _align_inputs(
        self,
        *,
        market_data: pd.DataFrame,
        labels_df: pd.DataFrame,
        probs_df: Optional[pd.DataFrame],
        economic_df: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """Align market data, labels, probabilities, and economic features."""
        # Ensure all indices are datetime and sorted
        def _prepare(df: pd.DataFrame) -> pd.DataFrame:
            if not isinstance(df.index, pd.DatetimeIndex):
                df = df.copy()
                df.index = pd.to_datetime(df.index)
            return df.sort_index()

        market_data = _prepare(market_data)
        labels_df = _prepare(labels_df)

        frames = [
            market_data[[col for col in market_data.columns if col.lower() in {"open", "high", "low", "close", "volume"}]].rename(
                columns=lambda c: c.lower()
            ),
            labels_df,
        ]

        if probs_df is not None and not probs_df.empty:
            probs_df = _prepare(probs_df)
            frames.append(probs_df)

        if economic_df is not None and not economic_df.empty:
            economic_df = _prepare(economic_df)
            frames.append(economic_df)

        # Inner join on time to ensure all required information is present
        aligned = frames[0]
        for extra in frames[1:]:
            aligned = aligned.join(extra, how="inner")

        aligned = aligned.dropna(how="all")

        tprint_info(
            f"🔗 Aligned dataset shape: {aligned.shape} "
            f"({aligned.index.min()} → {aligned.index.max()})"
        )

        return aligned

    def _compute_alpha_labels(
        self,
        aligned_df: pd.DataFrame,
        config: Dict[str, Any],
    ) -> pd.DataFrame:
        """Compute forward-return-based alpha labels on the aligned dataset.

        For regression:
            - Compute forward returns for horizons 1–4h.
            - Use the average of these horizons as a smoother macro alpha target.

        For classification:
            - Use the sign of the 1h forward return.
        """

        return_type = str(config.get("alpha_return_type", "log")).lower()
        target_type = str(config.get("alpha_target_type", "regression")).lower()

        df = aligned_df.copy()
        if "close" not in df.columns:
            raise ValueError("Aligned dataset must contain a 'close' column for returns")

        close = df["close"].astype(float)

        # Always compute 1h forward return
        if return_type == "simple":
            fwd_ret_1h = close.shift(-1) / close - 1.0
        else:
            fwd_ret_1h = np.log(close.shift(-1) / close)
        df["alpha_forward_return_1h"] = fwd_ret_1h

        # Multi-horizon forward returns (1–4h)
        max_h = 4
        fwd_cols: Dict[int, str] = {1: "alpha_forward_return_1h"}
        for h in range(2, max_h + 1):
            if return_type == "simple":
                fwd_ret_h = close.shift(-h) / close - 1.0
            else:
                fwd_ret_h = np.log(close.shift(-h) / close)
            col_name = f"alpha_forward_return_{h}h"
            df[col_name] = fwd_ret_h
            fwd_cols[h] = col_name

        if target_type == "regression":
            # Average of 1–4h forward returns as macro target
            horizon_keys = [h for h in fwd_cols.keys() if 1 <= h <= max_h]
            fwd_stack = [df[fwd_cols[h]] for h in horizon_keys]
            multi_target = pd.concat(fwd_stack, axis=1).mean(axis=1)
            df["alpha_target"] = multi_target
            effective_horizon = f"1-{max_h}h_mean"
        else:
            # Classification: sign of 1h forward return
            target = (fwd_ret_1h > 0).astype(float)
            target = target.where(~fwd_ret_1h.isna())
            df["alpha_target"] = target
            effective_horizon = "1h_classification"

        # Drop rows where we cannot compute all required forward returns
        required_cols = ["alpha_target"] + [fwd_cols[h] for h in sorted(fwd_cols.keys())]
        before = len(df)
        df = df.dropna(subset=required_cols)
        dropped = before - len(df)
        if dropped > 0:
            tprint_warning(
                f"Dropped {dropped} rows with NaN forward returns when building alpha labels"
            )

        tprint_info(
            f"🎯 Alpha label dataset shape: {df.shape} "
            f"(target_type={target_type}, effective_horizon={effective_horizon}, return_type={return_type})"
        )

        return df

    def _train_alpha_model(
        self,
        alpha_df: pd.DataFrame,
        config: Dict[str, Any],
    ) -> Tuple[Any, Optional[pd.Series], str, Dict[str, Any], Dict[str, Any]]:
        """Train a LightGBM model to predict alpha targets."""
        try:
            import lightgbm as lgb  # type: ignore[import]
        except ImportError as e:  # pragma: no cover - environment dependent
            raise ImportError("lightgbm is required for alpha model training") from e

        try:
            from sklearn.metrics import (
                accuracy_score,
                roc_auc_score,
                r2_score,
                mean_squared_error,
            )
        except ImportError:  # pragma: no cover - optional metrics
            accuracy_score = None  # type: ignore[assignment]
            roc_auc_score = None  # type: ignore[assignment]
            r2_score = None  # type: ignore[assignment]
            mean_squared_error = None  # type: ignore[assignment]

        def _safe_rmse(y_true: pd.Series, y_pred: np.ndarray) -> Optional[float]:
            if mean_squared_error is None:
                return None
            try:
                return float(mean_squared_error(y_true, y_pred, squared=False))
            except TypeError:
                return float(np.sqrt(mean_squared_error(y_true, y_pred)))  # type: ignore[call-arg]

        # Default to regression model for continuous alpha
        target_type = str(config.get("alpha_target_type", "regression")).lower()
        horizon = int(config.get("alpha_horizon_bars", 1))

        df = alpha_df.copy()
        if "alpha_target" not in df.columns:
            raise ValueError("alpha_target column not found in dataset")

        df = df.dropna(subset=["alpha_target"])
        if df.empty:
            raise ValueError("No valid samples for alpha model training after dropping NaNs")

        y = df["alpha_target"]

        numeric_df = df.select_dtypes(include=[np.number])
        feature_cols = [
            col
            for col in numeric_df.columns
            if col != "alpha_target" and not col.startswith("alpha_forward_return_")
        ]

        if not feature_cols:
            raise ValueError("No numeric features available for alpha model training")

        X = numeric_df[feature_cols]

        min_samples = int(config.get("alpha_min_samples", 200))
        if len(X) < max(min_samples, 20):
            raise ValueError(
                f"Insufficient samples for alpha model training: {len(X)} < {min_samples}"
            )

        train_frac = float(config.get("alpha_train_fraction", 0.8))
        train_frac = min(max(train_frac, 0.5), 0.95)
        split_idx = int(len(X) * train_frac)
        split_idx = max(min(split_idx, len(X) - 1), 1)

        # Chronological split to preserve temporal ordering
        X_train_raw, y_train = X.iloc[:split_idx].copy(), y.iloc[:split_idx]
        X_val_raw, y_val = X.iloc[split_idx:].copy(), y.iloc[split_idx:]

        # Robust scaling with optional outlier handling (no VectorBT to keep deps minimal)
        outlier_threshold = float(config.get("alpha_outlier_threshold", 3.0))
        normalizer_config: Dict[str, Any] = {
            "default_strategy": "robust",
            "auto_select": False,
            "handle_outliers": True,
            "outlier_threshold": outlier_threshold,
            "use_vectorbt": False,
        }
        scaler = ScalingNormalizer(normalizer_config)

        X_train_scaled = scaler.fit_transform(X_train_raw, strategy="robust")
        X_val_scaled = scaler.transform(X_val_raw)
        X_scaled_full = scaler.transform(X)

        # Optionally apply EWMA temporal smoothing on the scaled space
        use_ewm_features = bool(config.get("alpha_use_ewm_features", True))
        ewma_periods_cfg = config.get("alpha_ewm_periods", [2, 6, 10])
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

                    # apply_ewm_smoothing returns [original, ewm] for our usage;
                    # take only the EWM-smoothed block so dimensionality stays constant.
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
                # Fallback: use robust-scaled features without EWMA smoothing
                X_train = X_train_scaled
                X_val = X_val_scaled
                extended_feature_names = list(X_scaled_full.columns)
        else:
            X_train = X_train_scaled
            X_val = X_val_scaled
            extended_feature_names = list(X_scaled_full.columns)

        training_metrics: Dict[str, Any] = {}
        training_metrics["scaling_strategy"] = "robust"
        training_metrics["alpha_outlier_threshold"] = outlier_threshold
        training_metrics["alpha_use_ewm_features"] = use_ewm_features
        training_metrics["alpha_ewm_periods"] = ewma_periods

        # Prepare feature pipeline artifacts for persistence (feature names + scaler state)
        feature_pipeline_artifacts: Dict[str, Any] = {
            "feature_names": extended_feature_names,
            "scaler": scaler,
            "normalizer_config": normalizer_config,
        }

        # Optional hierarchical HPO for LightGBM hyperparameters (regression only, config-gated)
        best_hpo_params: Dict[str, Any] = {}
        enable_hpo = bool(config.get("alpha_enable_hpo", False)) and target_type == "regression"
        if enable_hpo:
            try:
                from src.utils.ml_common.optimization import default_objective_function
            except Exception as hpo_import_err:
                tprint_warning(f"Alpha HPO disabled due to import error: {hpo_import_err}")
                enable_hpo = False
            else:
                try:
                    hpo_param_groups = [
                        ParameterGroup(
                            name="lgbm_core",
                            params={
                                "num_leaves": {"type": "int", "low": 16, "high": 128},
                                "subsample": {"type": "float", "low": 0.5, "high": 1.0},
                                "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
                                "learning_rate": {"type": "float", "low": 0.01, "high": 0.2, "log": True},
                            },
                            priority=1,
                        ),
                    ]

                    base_model_for_hpo = lgb.LGBMRegressor(
                        n_estimators=int(config.get("alpha_n_estimators", 300)),
                        random_state=int(config.get("alpha_random_state", 42)),
                    )

                    optimizer = HierarchicalParameterOptimizer(
                        param_groups=hpo_param_groups,
                        objective_func=default_objective_function,
                        stages=[OptimizationStage.COARSE_GRID, OptimizationStage.TPE],
                        cv_folds=int(config.get("alpha_hpo_cv_folds", 3)),
                        scoring_metric="r2",
                        direction="maximize",
                        n_rounds=1,
                        enable_final_refinement=False,
                        final_refinement_trials=int(config.get("alpha_hpo_final_trials", 20)),
                        verbose=False,
                        use_custom_balanced_score=False,
                    )

                    X_train_hpo = X_train.to_numpy(dtype=float, copy=False)
                    y_train_hpo = y_train.to_numpy(dtype=float, copy=False)
                    X_val_hpo = X_val.to_numpy(dtype=float, copy=False) if len(X_val) > 0 else None
                    y_val_hpo = y_val.to_numpy(dtype=float, copy=False) if len(X_val) > 0 else None

                    hpo_result = optimizer.optimize(
                        X_train=X_train_hpo,
                        y_train=y_train_hpo,
                        X_val=X_val_hpo,
                        y_val=y_val_hpo,
                        model=base_model_for_hpo,
                    )

                    best_hpo_params = hpo_result.best_params or {}
                    training_metrics["alpha_hpo_best_score"] = float(hpo_result.best_score)
                    training_metrics["alpha_hpo_best_params"] = best_hpo_params
                    training_metrics["alpha_hpo_used"] = True
                except Exception as hpo_exc:
                    tprint_warning(f"Alpha HPO failed; proceeding with default hyperparameters: {hpo_exc}")
                    best_hpo_params = {}
                    training_metrics["alpha_hpo_used"] = False
        else:
            training_metrics["alpha_hpo_used"] = False

        if target_type == "regression":
            base_params: Dict[str, Any] = {
                "n_estimators": int(config.get("alpha_n_estimators", 300)),
                "learning_rate": float(config.get("alpha_learning_rate", 0.05)),
                "num_leaves": int(config.get("alpha_num_leaves", 64)),
                "subsample": float(config.get("alpha_subsample", 0.8)),
                "colsample_bytree": float(config.get("alpha_colsample_bytree", 0.8)),
                "random_state": int(config.get("alpha_random_state", 42)),
            }

            if best_hpo_params:
                for key, value in best_hpo_params.items():
                    if key in base_params:
                        base_params[key] = value

            model = lgb.LGBMRegressor(**base_params)
            model.fit(X_train, y_train)

            train_pred = model.predict(X_train)
            if r2_score is not None:
                training_metrics["train_r2"] = float(r2_score(y_train, train_pred))
            if mean_squared_error is not None:
                rmse_val = _safe_rmse(y_train, train_pred)
                if rmse_val is not None:
                    training_metrics["train_rmse"] = rmse_val

            if len(X_val) > 0:
                val_pred = model.predict(X_val)
                if r2_score is not None:
                    training_metrics["val_r2"] = float(r2_score(y_val, val_pred))
                if mean_squared_error is not None:
                    rmse_val = _safe_rmse(y_val, val_pred)
                    if rmse_val is not None:
                        training_metrics["val_rmse"] = rmse_val

            scores = pd.Series(model.predict(X_scaled_full), index=df.index, name="alpha_pred_return")
            pred_col_name = "alpha_pred_return"
            training_metrics["model_type"] = "lightgbm_regression"

        else:
            model = lgb.LGBMClassifier(
                n_estimators=int(config.get("alpha_n_estimators", 300)),
                learning_rate=float(config.get("alpha_learning_rate", 0.05)),
                num_leaves=int(config.get("alpha_num_leaves", 64)),
                subsample=float(config.get("alpha_subsample", 0.8)),
                colsample_bytree=float(config.get("alpha_colsample_bytree", 0.8)),
                random_state=int(config.get("alpha_random_state", 42)),
            )
            model.fit(X_train, y_train)

            train_proba = model.predict_proba(X_train)[:, 1]
            train_pred = (train_proba > 0.5).astype(float)

            if roc_auc_score is not None:
                training_metrics["train_auc"] = float(roc_auc_score(y_train, train_proba))
            if accuracy_score is not None:
                training_metrics["train_accuracy"] = float(accuracy_score(y_train, train_pred))

            if len(X_val) > 0:
                val_proba = model.predict_proba(X_val)[:, 1]
                val_pred = (val_proba > 0.5).astype(float)
                if roc_auc_score is not None:
                    training_metrics["val_auc"] = float(roc_auc_score(y_val, val_proba))
                if accuracy_score is not None:
                    training_metrics["val_accuracy"] = float(accuracy_score(y_val, val_pred))

            proba_all = model.predict_proba(X_scaled_full)[:, 1]
            scores = pd.Series(proba_all, index=df.index, name="alpha_pred_prob")
            pred_col_name = "alpha_pred_prob"
            training_metrics["model_type"] = "lightgbm_classification"

        full_scores = scores.reindex(alpha_df.index)

        # Optional SHAP analysis add-on (config-gated)
        if bool(config.get("alpha_enable_shap", False)):
            try:
                from src.utils.ml_common.explainability.model_explanations import (
                    explain_model_with_shap_lime,
                )

                X_train_arr = X_train.to_numpy(dtype=float, copy=False)
                if len(X_val) > 0:
                    X_test_arr = X_val.to_numpy(dtype=float, copy=False)
                else:
                    X_test_arr = X_train_arr

                shap_cfg: Dict[str, Any] = {
                    "enable_shap": True,
                    "enable_lime": False,
                    "shap_sample_size": int(config.get("alpha_shap_sample_size", 128)),
                }

                shap_results = explain_model_with_shap_lime(
                    model=model,
                    X_train=X_train_arr,
                    X_test=X_test_arr,
                    feature_names=extended_feature_names,
                    model_name="hmm_alpha_model",
                    config=shap_cfg,
                )

                shap_expl = shap_results.get("shap_explanations", {}) if isinstance(shap_results, dict) else {}
                if isinstance(shap_expl, dict):
                    training_metrics["alpha_shap_top_features"] = shap_expl.get("top_features", [])
                    training_metrics["alpha_shap_mean_importance"] = shap_expl.get("mean_importance")
            except Exception as shap_exc:
                tprint_warning(f"Alpha SHAP analysis failed (ignored): {shap_exc}")

        tprint_info(
            f"🤖 Trained LightGBM alpha model ({training_metrics.get('model_type', 'unknown')}) "
            f"on {len(X_train)} train / {len(X_val)} val samples"
        )

        training_metrics["alpha_horizon_bars"] = horizon
        training_metrics["target_type"] = target_type

        return model, full_scores, pred_col_name, training_metrics, feature_pipeline_artifacts

    def _calculate_iqr_winsorization_percentiles(
        self, data: pd.Series
    ) -> Tuple[float, float]:
        """Calculate winsorization percentiles using the IQR method.

        Returns fence values at Q1 - 1.5*IQR and Q3 + 1.5*IQR, then converts
        them to percentiles in the data distribution.

        Args:
            data: Series of numeric values

        Returns:
            Tuple of (lower_percentile, upper_percentile) for winsorization
        """
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1

        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr

        # Convert fence values to percentiles
        lower_pct = (data <= lower_fence).sum() / len(data)
        upper_pct = (data >= upper_fence).sum() / len(data)

        # Ensure we have reasonable bounds (at least 1%, at most 20%)
        lower_pct = max(0.01, min(lower_pct, 0.20))
        upper_pct = max(0.01, min(upper_pct, 0.20))

        return lower_pct, upper_pct

    def _calculate_winsorized_cv(
        self, data: pd.Series, lower_pct: Optional[float] = None, upper_pct: Optional[float] = None
    ) -> Tuple[float, float]:
        """Calculate both standard CV and Winsorized CV.

        If percentiles are not provided, uses IQR method to determine them.

        Args:
            data: Series of numeric values
            lower_pct: Lower percentile for winsorization (optional)
            upper_pct: Upper percentile for winsorization (optional)

        Returns:
            Tuple of (standard_cv, winsorized_cv)
        """
        data_clean = data.dropna()
        if len(data_clean) < 2:
            return 0.0, 0.0

        # Standard CV
        mean_val = data_clean.mean()
        std_val = data_clean.std()
        cv = abs(std_val / (mean_val + 1e-8))

        # Winsorized CV using IQR method if percentiles not provided
        if lower_pct is None or upper_pct is None:
            lower_pct, upper_pct = self._calculate_iqr_winsorization_percentiles(data_clean)

        lower_bound = data_clean.quantile(lower_pct)
        upper_bound = data_clean.quantile(1.0 - upper_pct)

        # Winsorize: cap values at bounds
        winsorized = data_clean.clip(lower=lower_bound, upper=upper_bound)

        win_mean = winsorized.mean()
        win_std = winsorized.std()
        wcv = abs(win_std / (win_mean + 1e-8))

        return cv, wcv

    def _calculate_economic_cv_ratio(
        self,
        regime_labels: np.ndarray,
        forward_returns: pd.Series,
        use_winsorized: bool = True,
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate Economic CV Ratio: Between-Regime CV / Within-Regime CV.

        This measures signal-to-noise: how distinct regime means are (signal)
        versus how noisy returns are within each regime (noise).

        Args:
            regime_labels: Array of regime assignments
            forward_returns: Series of forward returns aligned with labels
            use_winsorized: Whether to use winsorized CV (default: True)

        Returns:
            Tuple of (cv_ratio, metrics_dict) where metrics_dict contains:
                - between_cv / between_wcv
                - within_cv / within_wcv (sample-size weighted)
                - cv_ratio / wcv_ratio
        """
        unique_regimes = np.unique(regime_labels)
        if len(unique_regimes) < 2:
            return 0.0, {}

        # Calculate per-regime statistics
        regime_means = []
        regime_sizes = []
        within_cvs = []
        within_wcvs = []

        for regime in unique_regimes:
            mask = regime_labels == regime
            regime_returns = forward_returns[mask].dropna()

            if len(regime_returns) < 2:
                continue

            regime_mean = regime_returns.mean()
            regime_means.append(regime_mean)
            regime_sizes.append(len(regime_returns))

            # Within-regime CV and WCV
            cv, wcv = self._calculate_winsorized_cv(regime_returns)
            within_cvs.append(cv)
            within_wcvs.append(wcv)

        if len(regime_means) < 2:
            return 0.0, {}

        # Between-Regime CV: variability of regime means
        regime_means_series = pd.Series(regime_means)
        between_mean = regime_means_series.mean()
        between_std = regime_means_series.std()
        between_cv = abs(between_std / (between_mean + 1e-8))

        # For between WCV, we winsorize the regime means themselves
        _, between_wcv = self._calculate_winsorized_cv(regime_means_series)

        # Within-Regime CV: sample-size weighted average of within-regime CVs
        total_samples = sum(regime_sizes)
        weights = [size / total_samples for size in regime_sizes]

        within_cv_weighted = sum(cv * w for cv, w in zip(within_cvs, weights))
        within_wcv_weighted = sum(wcv * w for wcv, w in zip(within_wcvs, weights))

        # CV Ratio: signal-to-noise
        cv_ratio = between_cv / (within_cv_weighted + 1e-8)
        wcv_ratio = between_wcv / (within_wcv_weighted + 1e-8)

        metrics = {
            "between_cv": between_cv,
            "between_wcv": between_wcv,
            "within_cv": within_cv_weighted,
            "within_wcv": within_wcv_weighted,
            "cv_ratio": cv_ratio,
            "wcv_ratio": wcv_ratio,
            "n_regimes": len(unique_regimes),
            "total_samples": total_samples,
        }

        # Use winsorized ratio if requested
        final_ratio = wcv_ratio if use_winsorized else cv_ratio

        return final_ratio, metrics

    def _optimize_regime_boundaries(
        self,
        alpha_scores: pd.Series,
        forward_returns: pd.Series,
        n_regimes: int,
        min_bin_pct: float = 0.10,
        max_bin_pct: float = 0.35,
        jiggle_pct: float = 0.01,
        max_iterations: int = 100,
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """Optimize regime boundaries to maximize Economic CV Ratio.

        Uses iterative "jiggle" search: starts with quantile boundaries,
        then systematically shifts each boundary to improve the objective.

        Args:
            alpha_scores: Predicted alpha scores (feature)
            forward_returns: Forward returns (target)
            n_regimes: Number of regimes to create
            min_bin_pct: Minimum percentage of samples per bin (default: 10%)
            max_bin_pct: Maximum percentage of samples per bin (default: 35%)
            jiggle_pct: Percentage of data to shift boundaries by (default: 1%)
            max_iterations: Maximum optimization iterations (default: 100)

        Returns:
            Tuple of (optimal_labels, best_score, metrics_dict)
        """
        # Sort data by alpha scores
        sorted_indices = alpha_scores.argsort()
        sorted_scores = alpha_scores.iloc[sorted_indices].values
        sorted_returns = forward_returns.iloc[sorted_indices]

        n_samples = len(sorted_scores)
        min_bin_size = int(n_samples * min_bin_pct)
        max_bin_size = int(n_samples * max_bin_pct)
        jiggle_size = max(1, int(n_samples * jiggle_pct))

        # Initialize with quantile boundaries
        quantile_indices = [int(n_samples * i / n_regimes) for i in range(1, n_regimes)]
        current_boundaries = np.array(quantile_indices, dtype=int)

        # Ensure boundaries respect min/max bin size
        current_boundaries = self._enforce_bin_constraints(
            current_boundaries, n_samples, min_bin_size, max_bin_size
        )

        # Calculate initial score
        current_labels = self._boundaries_to_labels(current_boundaries, n_samples)
        current_score, current_metrics = self._calculate_economic_cv_ratio(
            current_labels, sorted_returns, use_winsorized=True
        )

        best_boundaries = current_boundaries.copy()
        best_score = current_score
        best_metrics = current_metrics

        improved = True
        iteration = 0

        while improved and iteration < max_iterations:
            improved = False
            iteration += 1

            # Try jiggling each boundary
            for i in range(len(current_boundaries)):
                # Try moving left
                test_boundaries = current_boundaries.copy()
                test_boundaries[i] = max(
                    test_boundaries[i] - jiggle_size,
                    test_boundaries[i - 1] + min_bin_size if i > 0 else min_bin_size
                )

                # Check if this configuration satisfies constraints
                if self._check_bin_constraints(test_boundaries, n_samples, min_bin_size, max_bin_size):
                    test_labels = self._boundaries_to_labels(test_boundaries, n_samples)
                    test_score, test_metrics = self._calculate_economic_cv_ratio(
                        test_labels, sorted_returns, use_winsorized=True
                    )

                    if test_score > best_score:
                        best_boundaries = test_boundaries.copy()
                        best_score = test_score
                        best_metrics = test_metrics
                        improved = True

                # Try moving right
                test_boundaries = current_boundaries.copy()
                test_boundaries[i] = min(
                    test_boundaries[i] + jiggle_size,
                    (test_boundaries[i + 1] - min_bin_size if i < len(current_boundaries) - 1
                     else n_samples - min_bin_size)
                )

                if self._check_bin_constraints(test_boundaries, n_samples, min_bin_size, max_bin_size):
                    test_labels = self._boundaries_to_labels(test_boundaries, n_samples)
                    test_score, test_metrics = self._calculate_economic_cv_ratio(
                        test_labels, sorted_returns, use_winsorized=True
                    )

                    if test_score > best_score:
                        best_boundaries = test_boundaries.copy()
                        best_score = test_score
                        best_metrics = test_metrics
                        improved = True

            if improved:
                current_boundaries = best_boundaries.copy()
                current_score = best_score

        # Convert sorted labels back to original order
        optimal_labels_sorted = self._boundaries_to_labels(best_boundaries, n_samples)
        optimal_labels = np.empty(n_samples, dtype=int)
        optimal_labels[sorted_indices] = optimal_labels_sorted

        best_metrics["optimization_iterations"] = iteration
        best_metrics["converged"] = not improved or iteration < max_iterations

        return optimal_labels, best_score, best_metrics

    def _enforce_bin_constraints(
        self,
        boundaries: np.ndarray,
        n_samples: int,
        min_bin_size: int,
        max_bin_size: int,
    ) -> np.ndarray:
        """Enforce minimum and maximum bin size constraints on boundaries."""
        adjusted = boundaries.copy()

        # Ensure first bin is large enough
        adjusted[0] = max(adjusted[0], min_bin_size)

        # Ensure spacing between boundaries
        for i in range(1, len(adjusted)):
            adjusted[i] = max(adjusted[i], adjusted[i-1] + min_bin_size)

        # Ensure last bin is large enough
        adjusted[-1] = min(adjusted[-1], n_samples - min_bin_size)

        return adjusted

    def _check_bin_constraints(
        self,
        boundaries: np.ndarray,
        n_samples: int,
        min_bin_size: int,
        max_bin_size: int,
    ) -> bool:
        """Check if boundaries satisfy bin size constraints."""
        # Check first bin
        if boundaries[0] < min_bin_size or boundaries[0] > max_bin_size:
            return False

        # Check middle bins
        for i in range(1, len(boundaries)):
            bin_size = boundaries[i] - boundaries[i-1]
            if bin_size < min_bin_size or bin_size > max_bin_size:
                return False

        # Check last bin
        last_bin_size = n_samples - boundaries[-1]
        if last_bin_size < min_bin_size or last_bin_size > max_bin_size:
            return False

        return True

    def _boundaries_to_labels(self, boundaries: np.ndarray, n_samples: int) -> np.ndarray:
        """Convert boundary indices to regime labels."""
        labels = np.zeros(n_samples, dtype=int)

        for i, boundary in enumerate(boundaries):
            labels[boundary:] = i + 1

        return labels

    def _assign_alpha_regimes(
        self,
        alpha_df: pd.DataFrame,
        alpha_scores: pd.Series,
        config: Dict[str, Any],
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[str]]:
        """Derive alpha regimes using flexible quantile optimization.

        This method now:
        1. Tests multiple regime counts (4, 5, 6)
        2. Optimizes boundaries to maximize Economic CV Ratio
        3. Enforces 10-35% bin size constraints
        4. Uses sample-size weighted Within-CV
        5. Reports both CV and Winsorized CV metrics
        """
        # Get configuration for regime optimization
        use_flexible_regimes = bool(config.get("alpha_use_flexible_regimes", True))
        regime_counts_to_test = config.get("alpha_regime_counts", [4, 5, 6])
        min_bin_pct = float(config.get("alpha_min_bin_pct", 0.10))
        max_bin_pct = float(config.get("alpha_max_bin_pct", 0.35))

        # Validate regime counts
        if isinstance(regime_counts_to_test, int):
            regime_counts_to_test = [regime_counts_to_test]
        regime_counts_to_test = [n for n in regime_counts_to_test if 3 <= n <= 6]
        if not regime_counts_to_test:
            regime_counts_to_test = [5]  # Fallback to default

        # Optional smoothing of alpha scores before constructing regimes
        score_smoothing_method = str(
            config.get("alpha_score_smoothing_method", "none")
        ).lower()
        score_smoothing_window = int(config.get("alpha_score_smoothing_window", 1))
        scores_for_binning = alpha_scores.copy()
        if score_smoothing_method != "none" and score_smoothing_window > 1:
            try:
                if score_smoothing_method == "ewm":
                    scores_for_binning = scores_for_binning.ewm(
                        span=score_smoothing_window, adjust=False
                    ).mean()
                elif score_smoothing_method == "rolling_median":
                    scores_for_binning = (
                        scores_for_binning
                        .rolling(window=score_smoothing_window, min_periods=1)
                        .median()
                    )
                elif score_smoothing_method == "rolling_mean":
                    scores_for_binning = (
                        scores_for_binning
                        .rolling(window=score_smoothing_window, min_periods=1)
                        .mean()
                    )
                tprint_info(
                    f"🧹 Applied {score_smoothing_method} smoothing to alpha scores "
                    f"(window={score_smoothing_window}) before regime binning"
                )
            except Exception as score_smooth_exc:
                tprint_warning(
                    "Alpha score smoothing failed (ignored, using raw scores): "
                    f"{score_smooth_exc}"
                )
                scores_for_binning = alpha_scores

        valid_scores = scores_for_binning.dropna()
        if valid_scores.empty:
            tprint_warning(
                f"Not enough valid alpha scores ({len(valid_scores)}) to define regimes"
            )
            return alpha_df, None, None

        # Get forward returns for optimization
        fwd_cols = [col for col in alpha_df.columns if col.startswith("alpha_forward_return_")]
        if not fwd_cols:
            tprint_warning("No alpha_forward_return column found for regime optimization")
            return alpha_df, None, None

        fwd_col = fwd_cols[0]
        forward_returns = alpha_df[fwd_col].dropna()

        # Align scores and returns
        common_idx = valid_scores.index.intersection(forward_returns.index)
        if len(common_idx) < 20:
            tprint_warning(
                f"Not enough valid samples ({len(common_idx)}) for regime optimization"
            )
            return alpha_df, None, None

        aligned_scores = valid_scores.loc[common_idx]
        aligned_returns = forward_returns.loc[common_idx]

        # Multi-regime competition: test multiple regime counts
        if use_flexible_regimes:
            tprint_info(
                f"🔍 Testing flexible regime optimization for {regime_counts_to_test} regime counts "
                f"(min_bin={min_bin_pct*100:.0f}%, max_bin={max_bin_pct*100:.0f}%)"
            )

            best_n_regimes = None
            best_score = -np.inf
            best_labels = None
            best_metrics = None
            competition_results = []

            for n_regimes in regime_counts_to_test:
                if len(aligned_scores) < n_regimes * int(len(aligned_scores) * min_bin_pct):
                    tprint_warning(
                        f"Skipping {n_regimes} regimes: insufficient samples for bin constraints"
                    )
                    continue

                try:
                    labels, score, metrics = self._optimize_regime_boundaries(
                        alpha_scores=aligned_scores,
                        forward_returns=aligned_returns,
                        n_regimes=n_regimes,
                        min_bin_pct=min_bin_pct,
                        max_bin_pct=max_bin_pct,
                        jiggle_pct=float(config.get("alpha_jiggle_pct", 0.01)),
                        max_iterations=int(config.get("alpha_max_opt_iterations", 100)),
                    )

                    competition_results.append({
                        "n_regimes": n_regimes,
                        "wcv_ratio": score,
                        "cv_ratio": metrics.get("cv_ratio", 0.0),
                        "between_cv": metrics.get("between_cv", 0.0),
                        "between_wcv": metrics.get("between_wcv", 0.0),
                        "within_cv": metrics.get("within_cv", 0.0),
                        "within_wcv": metrics.get("within_wcv", 0.0),
                        "iterations": metrics.get("optimization_iterations", 0),
                        "converged": metrics.get("converged", False),
                    })

                    if score > best_score:
                        best_score = score
                        best_n_regimes = n_regimes
                        best_labels = labels
                        best_metrics = metrics

                    tprint_info(
                        f"  → {n_regimes} regimes: WCV Ratio={score:.4f}, "
                        f"CV Ratio={metrics.get('cv_ratio', 0.0):.4f}, "
                        f"iterations={metrics.get('optimization_iterations', 0)}"
                    )

                except Exception as opt_exc:
                    tprint_warning(f"Optimization failed for {n_regimes} regimes: {opt_exc}")
                    continue

            if best_labels is None:
                tprint_warning("All regime optimization attempts failed; falling back to simple quantiles")
                use_flexible_regimes = False
            else:
                num_bins = best_n_regimes
                bucket_codes = pd.Series(best_labels, index=common_idx)
                tprint_info(
                    f"✅ Selected {num_bins} regimes with WCV Ratio={best_score:.4f} "
                    f"(Between WCV={best_metrics.get('between_wcv', 0.0):.4f}, "
                    f"Within WCV={best_metrics.get('within_wcv', 0.0):.4f})"
                )

        # Fallback to simple quantile binning if flexible optimization is disabled or failed
        if not use_flexible_regimes:
            num_bins = int(config.get("alpha_regime_bins", 5))
            num_bins = max(3, min(num_bins, 6))

            if len(aligned_scores) < num_bins:
                if len(aligned_scores) >= 3:
                    num_bins = len(aligned_scores)
                    tprint_warning(
                        f"Reducing alpha regime bins to {num_bins} due to limited samples"
                    )
                else:
                    tprint_warning(
                        f"Not enough valid alpha scores ({len(aligned_scores)}) to define regimes"
                    )
                    return alpha_df, None, None

            try:
                ranks = aligned_scores.rank(method="first")
                bucket_codes = pd.qcut(ranks, q=num_bins, labels=False)
                tprint_info(
                    f"📊 Using simple quantile binning with {num_bins} equal-sized regimes"
                )
            except ValueError as e:
                tprint_warning(f"Failed to compute quantile-based alpha regimes: {e}")
                return alpha_df, None, None

        bucket_col = f"alpha_regime_bucket_{num_bins}"
        alpha_df[bucket_col] = bucket_codes.reindex(alpha_df.index)

        # Compute comprehensive regime statistics with CV and WCV metrics
        fwd_col = fwd_cols[0]

        stats_records = []
        for bucket in sorted(bucket_codes.unique()):
            mask = alpha_df[bucket_col] == bucket
            group = alpha_df.loc[mask]
            if group.empty:
                continue

            ret = group[fwd_col]
            if ret.isna().all():
                continue

            ret = ret.astype(float)
            mean_ret = float(ret.mean())
            std_ret = float(ret.std()) if len(ret) > 1 else 0.0
            sharpe = mean_ret / (std_ret + 1e-8) if std_ret > 0 else 0.0

            # CV and Winsorized CV for this regime
            cv, wcv = self._calculate_winsorized_cv(ret)

            # Realized volatility (same units as std_ret, kept separate for clarity)
            realized_vol = std_ret

            # Downside volatility: std of negative returns only
            downside = ret[ret < 0.0]
            downside_vol = float(downside.std()) if len(downside) > 1 else 0.0
            downside_sharpe = mean_ret / (downside_vol + 1e-8) if downside_vol > 0 else 0.0

            # Higher-moment and tail metrics
            skewness = float(ret.skew()) if len(ret) > 2 else 0.0
            kurtosis = float(ret.kurtosis()) if len(ret) > 3 else 0.0

            var_level = float(config.get("alpha_var_quantile", 0.05))
            var_level = min(max(var_level, 0.001), 0.2)
            try:
                var_ret = float(ret.quantile(var_level))
            except Exception:
                var_ret = 0.0
            downside_slice = ret[ret <= var_ret]
            cvar_ret = float(downside_slice.mean()) if len(downside_slice) > 0 else var_ret

            # Tail ratio: right tail vs left tail magnitude
            try:
                q_low = float(ret.quantile(var_level))
                q_high = float(ret.quantile(1.0 - var_level))
                right_tail = ret[ret >= q_high]
                left_tail = ret[ret <= q_low]
                right_mean = float(right_tail.mean()) if len(right_tail) > 0 else 0.0
                left_mean = float(left_tail.mean()) if len(left_tail) > 0 else 0.0
                tail_ratio = right_mean / (abs(left_mean) + 1e-8) if right_mean != 0.0 or left_mean != 0.0 else 0.0
            except Exception:
                tail_ratio = 0.0

            # Vol-of-vol: dispersion of rolling volatility within the regime
            vol_of_vol_window = int(config.get("alpha_vol_of_vol_window", 10))
            if vol_of_vol_window >= 2 and len(ret) >= vol_of_vol_window:
                rolling_vol = ret.rolling(vol_of_vol_window).std()
                vol_of_vol = float(rolling_vol.std(skipna=True))
            else:
                vol_of_vol = 0.0

            # Trendiness score: R^2 of linear fit on cumulative returns
            if len(ret) > 2:
                try:
                    t_idx = np.arange(len(ret), dtype=float)
                    cum_ret = ret.cumsum().to_numpy()
                    t_mean = float(t_idx.mean())
                    y_mean = float(cum_ret.mean())
                    cov_ty = float(np.dot(t_idx - t_mean, cum_ret - y_mean))
                    var_t = float(np.dot(t_idx - t_mean, t_idx - t_mean))
                    if var_t > 0.0:
                        slope = cov_ty / var_t
                        intercept = y_mean - slope * t_mean
                        y_hat = intercept + slope * t_idx
                        ss_res = float(np.sum((cum_ret - y_hat) ** 2))
                        ss_tot = float(np.sum((cum_ret - y_mean) ** 2))
                        trendiness = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
                    else:
                        trendiness = 0.0
                except Exception:
                    trendiness = 0.0
            else:
                trendiness = 0.0

            hit_rate = float((ret > 0).mean())
            mean_target = float(group["alpha_target"].mean())

            # Calculate bin percentage
            bin_pct = float(len(group)) / float(len(alpha_df))

            stats_records.append(
                {
                    "alpha_regime_bucket": int(bucket),
                    "n_samples": int(len(group)),
                    "bin_percentage": bin_pct,
                    "mean_forward_return": mean_ret,
                    "std_forward_return": std_ret,
                    "cv_forward_return": cv,
                    "wcv_forward_return": wcv,
                    "sharpe_forward_return": sharpe,
                    "realized_vol_forward_return": realized_vol,
                    "downside_vol_forward_return": downside_vol,
                    "downside_sharpe_forward_return": downside_sharpe,
                    "skewness_forward_return": skewness,
                    "kurtosis_forward_return": kurtosis,
                    "tail_ratio_forward_return": tail_ratio,
                    "var_forward_return": var_ret,
                    "cvar_forward_return": cvar_ret,
                    "vol_of_vol_forward_return": vol_of_vol,
                    "trendiness_forward_return": trendiness,
                    "hit_rate_positive_return": hit_rate,
                    "mean_alpha_target": mean_target,
                }
            )

        if not stats_records:
            tprint_warning("No stats records generated for alpha regimes")
            return alpha_df, None, bucket_col

        regime_stats_df = pd.DataFrame(stats_records).set_index("alpha_regime_bucket").sort_index()

        # Calculate overall economic metrics (Between/Within CV ratios)
        if use_flexible_regimes and best_metrics is not None:
            overall_metrics = {
                "overall_between_cv": best_metrics.get("between_cv", 0.0),
                "overall_between_wcv": best_metrics.get("between_wcv", 0.0),
                "overall_within_cv": best_metrics.get("within_cv", 0.0),
                "overall_within_wcv": best_metrics.get("within_wcv", 0.0),
                "overall_cv_ratio": best_metrics.get("cv_ratio", 0.0),
                "overall_wcv_ratio": best_metrics.get("wcv_ratio", 0.0),
                "optimization_iterations": best_metrics.get("optimization_iterations", 0),
                "optimization_converged": best_metrics.get("converged", False),
            }
            # Add as a row in the stats dataframe for easy reporting
            for key, value in overall_metrics.items():
                regime_stats_df[key] = value

        tprint_info(
            f"📊 Computed alpha regime statistics for {len(regime_stats_df)} regimes "
            f"(bins={num_bins})"
        )

        if use_flexible_regimes and best_metrics is not None:
            tprint_info(
                f"📈 Overall Economic Metrics: "
                f"CV Ratio={best_metrics.get('cv_ratio', 0.0):.4f}, "
                f"WCV Ratio={best_metrics.get('wcv_ratio', 0.0):.4f}, "
                f"Between WCV={best_metrics.get('between_wcv', 0.0):.4f}, "
                f"Within WCV={best_metrics.get('within_wcv', 0.0):.4f}"
            )

        return alpha_df, regime_stats_df, bucket_col

    def _assess_alpha_regime_quality(
        self,
        *,
        alpha_df: pd.DataFrame,
        regime_col: Optional[str],
        config: Dict[str, Any],
    ) -> Tuple[Optional[ClusterQualityMetrics], Optional[str]]:
        """Assess quality of alpha regimes using ClusterQualityAssessor.

        This uses the unified assess_quality interface and persists metrics as a
        dedicated artifact. The minimum regime size defaults to 3 but can be
        overridden via config["alpha_min_regime_size"].
        """
        if regime_col is None or regime_col not in alpha_df.columns:
            tprint_warning("No alpha regime column provided; skipping regime quality assessment")
            return None, None

        regime_series = alpha_df[regime_col]
        valid_mask = regime_series.notna()
        if valid_mask.sum() == 0:
            tprint_warning("No valid alpha regime labels for quality assessment")
            return None, None

        regime_labels = np.asarray(regime_series[valid_mask].astype(int), dtype=int)

        numeric_df = alpha_df.select_dtypes(include=[np.number])
        drop_cols = ["alpha_target", regime_col]
        drop_cols.extend([c for c in numeric_df.columns if c.startswith("alpha_forward_return_")])
        feature_cols = [c for c in numeric_df.columns if c not in drop_cols]
        if not feature_cols:
            forward_ret_feature_cols = [
                c for c in numeric_df.columns if c.startswith("alpha_forward_return_")
            ]
            if forward_ret_feature_cols:
                feature_cols = forward_ret_feature_cols
            else:
                tprint_warning("No numeric features available for alpha regime quality assessment")
                return None, None

        feature_data = numeric_df[feature_cols].loc[valid_mask]

        forward_ret_cols = [c for c in alpha_df.columns if c.startswith("alpha_forward_return_")]
        forward_returns = None
        if forward_ret_cols:
            forward_returns = alpha_df[forward_ret_cols[0]].loc[valid_mask]

        timestamps = alpha_df.index[valid_mask]
        min_regime_size = int(config.get("alpha_min_regime_size", 3))
        temporal_mode = str(config.get("alpha_temporal_sensitivity_mode", "regime_persistence_focused"))
        fast_mode = bool(config.get("alpha_quality_fast_mode", False))

        try:
            metrics = self.quality_assessor.assess_quality(
                regime_labels=regime_labels,
                feature_data=feature_data,
                forward_returns=forward_returns,
                timestamps=timestamps,
                min_regime_size=min_regime_size,
                temporal_sensitivity_mode=temporal_mode,
                fast_mode=fast_mode,
                standardize_for_metrics=True,
            )
        except Exception as exc:
            tprint_warning(f"Alpha regime quality assessment failed: {exc}")
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
                artifact_name="hmm_alpha_regime_quality_1h",
                artifact_type="data",
                metadata={
                    "min_regime_size": min_regime_size,
                },
            )
        except Exception as save_exc:
            tprint_warning(f"Failed to save alpha regime quality artifact: {save_exc}")
            quality_path = None

        return metrics, quality_path
