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

        Current implementation:
            - Computes forward 1-bar log returns on 'close'.
            - Builds a binary classification target: 1 if return > 0, else 0.
        """
        horizon = int(config.get("alpha_horizon_bars", 1))
        return_type = str(config.get("alpha_return_type", "log")).lower()
        # Default to regression so alpha_target is a continuous forward return
        target_type = str(config.get("alpha_target_type", "regression")).lower()

        df = aligned_df.copy()

        if "close" not in df.columns:
            raise ValueError("Aligned dataset must contain a 'close' column for returns")

        close = df["close"].astype(float)

        if return_type == "simple":
            fwd_ret = close.shift(-horizon) / close - 1.0
        else:
            # Default to log returns
            fwd_ret = np.log(close.shift(-horizon) / close)

        df[f"alpha_forward_return_{horizon}h"] = fwd_ret

        if target_type == "regression":
            # For regression, just use the forward return as target
            df["alpha_target"] = df[f"alpha_forward_return_{horizon}h"]
        else:
            # For classification, simple sign-based target
            target = (fwd_ret > 0).astype(float)
            target = target.where(~fwd_ret.isna())
            df["alpha_target"] = target

        # Drop rows where we cannot compute forward returns (usually the last few)
        before = len(df)
        df = df[df[f"alpha_forward_return_{horizon}h"].notna()]
        dropped = before - len(df)
        if dropped > 0:
            tprint_warning(
                f"Dropped {dropped} rows with NaN forward returns when building alpha labels"
            )

        tprint_info(
            f"🎯 Alpha label dataset shape: {df.shape} "
            f"(horizon={horizon}, return_type={return_type}, target_type={target_type})"
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

        # Optionally augment with EWMA-smoothed features on the scaled space
        use_ewm_features = bool(config.get("alpha_use_ewm_features", True))
        ewma_periods_cfg = config.get("alpha_ewm_periods", [2, 4, 6, 8])
        try:
            ewma_periods = [int(p) for p in ewma_periods_cfg if int(p) > 0]
        except Exception:
            ewma_periods = [2, 4, 6, 8]

        if use_ewm_features and ewma_periods:
            base_df = X_scaled_full.copy()
            feature_names_seq: List[str] = list(base_df.columns)
            features_df = base_df

            for period in ewma_periods:
                alpha_val = 2.0 / float(period + 1)
                try:
                    smoothed_array, feature_names_seq = apply_ewm_smoothing(
                        features_df.values,
                        alpha=alpha_val,
                        feature_names=feature_names_seq,
                        use_vectorization_optimization=False,
                    )
                    features_df = pd.DataFrame(
                        smoothed_array,
                        index=base_df.index,
                        columns=pd.Index(feature_names_seq),
                    )
                except Exception as e:
                    tprint_warning(f"EWMA feature generation failed for period={period}: {e}")
                    features_df = base_df
                    feature_names_seq = list(base_df.columns)
                    break

            X_features_full = features_df
            X_train = X_features_full.iloc[:split_idx].copy()
            X_val = X_features_full.iloc[split_idx:].copy()
            X_scaled_full = X_features_full
            extended_feature_names = feature_names_seq
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
                training_metrics["train_rmse"] = float(
                    mean_squared_error(y_train, train_pred, squared=False)
                )

            if len(X_val) > 0:
                val_pred = model.predict(X_val)
                if r2_score is not None:
                    training_metrics["val_r2"] = float(r2_score(y_val, val_pred))
                if mean_squared_error is not None:
                    training_metrics["val_rmse"] = float(
                        mean_squared_error(y_val, val_pred, squared=False)
                    )

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

    def _assign_alpha_regimes(
        self,
        alpha_df: pd.DataFrame,
        alpha_scores: pd.Series,
        config: Dict[str, Any],
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[str]]:
        """Derive 3–5 alpha regimes from predicted scores and compute stats."""
        # Default to 5 quantile bins over expected forward return
        num_bins = int(config.get("alpha_regime_bins", 5))
        if num_bins < 3:
            num_bins = 3
        if num_bins > 5:
            num_bins = 5

        valid_scores = alpha_scores.dropna()
        if valid_scores.empty or len(valid_scores) < num_bins:
            tprint_warning(
                f"Not enough valid alpha scores ({len(valid_scores)}) to define {num_bins} regimes"
            )
            return alpha_df, None, None

        try:
            ranks = valid_scores.rank(method="first")
            bucket_codes = pd.qcut(ranks, q=num_bins, labels=False)
        except ValueError as e:
            tprint_warning(f"Failed to compute quantile-based alpha regimes: {e}")
            return alpha_df, None, None

        bucket_col = f"alpha_regime_bucket_{num_bins}"
        alpha_df[bucket_col] = bucket_codes.reindex(alpha_df.index)

        fwd_cols = [col for col in alpha_df.columns if col.startswith("alpha_forward_return_")]
        if not fwd_cols:
            tprint_warning("No alpha_forward_return column found for regime statistics")
            return alpha_df, None, bucket_col

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

            stats_records.append(
                {
                    "alpha_regime_bucket": int(bucket),
                    "n_samples": int(len(group)),
                    "mean_forward_return": mean_ret,
                    "std_forward_return": std_ret,
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

        tprint_info(
            f"📊 Computed alpha regime statistics for {len(regime_stats_df)} regimes "
            f"(bins={num_bins})"
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
