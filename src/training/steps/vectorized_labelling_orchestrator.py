# src/training/steps/vectorized_labelling_orchestrator.py

"""Vectorized Labelling Orchestrator for comprehensive feature engineering and labeling pipeline.
Coordinates optimized_triple_barrier_labeling.py, vectorized_advanced_feature_engineering.py
and autoencoder_feature_generator.py with advanced preprocessing and feature selection.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger
from src.training.hmm_regime_barrier_optimizer import HMMRegimeBarrierOptimizer
from src.training.steps.step4_analyst_labeling_feature_engineering_components.regime_aware_triple_barrier_labeling import apply_regime_aware_triple_barrier_labeling_with_barriers


# -----------------------------------------------------------------------------
# Warnings logging setup
# -----------------------------------------------------------------------------
warnings.simplefilter("default")
_warning_logger = logging.getLogger("Ares.Warnings")
if not _warning_logger.handlers:
    try:
        fh = logging.FileHandler("log/python_warnings.log")
        fh.setLevel(logging.WARNING)
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(fmt)
        _warning_logger.addHandler(fh)
        _warning_logger.propagate = False
    except Exception:
        pass



warnings.showwarning = _showwarning


# -----------------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------------
class VectorizedLabellingOrchestrator:
    """Comprehensive vectorized labeling orchestrator that coordinates all feature generation
    and labeling components with advanced preprocessing and feature selection.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("VectorizedLabellingOrchestrator")
        self.feature_error_logger = logging.getLogger("Ares.FeatureError")
        if not self.feature_error_logger.handlers:
            try:
                fh = logging.FileHandler("log/feature_errors.log")
                fh.setLevel(logging.INFO)
                fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
                fh.setFormatter(fmt)
                self.feature_error_logger.addHandler(fh)
                self.feature_error_logger.propagate = False
            except Exception:
                pass

        # Configuration
        self.orchestrator_config = config.get("vectorized_labelling_orchestrator", {})
        self.enable_stationary_checks = self.orchestrator_config.get(
            "enable_stationary_checks", True
        )
        self.enable_data_normalization = self.orchestrator_config.get(
            "enable_data_normalization", True
        )
        self.enable_lookahead_bias_handling = self.orchestrator_config.get(
            "enable_lookahead_bias_handling", True
        )
        self.enable_feature_selection = self.orchestrator_config.get(
            "enable_feature_selection", True
        )
        self.enable_memory_efficient_types = self.orchestrator_config.get(
            "enable_memory_efficient_types", True
        )
        self.enable_parquet_saving = self.orchestrator_config.get(
            "enable_parquet_saving", True
        )
        # Auto HMM barrier recalculation for step4 labeling
        self.auto_recalculate_hmm_barriers = bool(
            self.orchestrator_config.get("auto_recalculate_hmm_barriers", True)
        )
        self.hmm_barrier_regime_column = str(
            self.orchestrator_config.get("hmm_barrier_regime_column", "hmm_regime")
        )
        # Strict feature shapes mode: treat scalar features as errors
        self.strict_feature_shapes = bool(
            self.orchestrator_config.get("strict_feature_shapes", True)
            or os.getenv("CI") == "1"
        )
        # NEW: Context/baseline column configuration
        self.keep_close_returns = self.orchestrator_config.get(
            "keep_only_close_returns_main", True
        )
        # Options: 'returns' (volume_returns), 'log', 'detrended', 'normalized', or 'none'
        self.volume_representation = str(
            self.orchestrator_config.get("volume_representation", "returns")
        ).lower()
        # Columns that should never be used as ML features (preserved as context)
        self.context_non_feature_columns: set[str] = set(
            self.orchestrator_config.get(
                "context_non_feature_columns",
                [
                    "year",
                    "month",
                    "day",
                    "exchange",
                    "symbol",
                    "timeframe",
                    "timestamp",
                    # Include selected raw info columns to carry as context (not features)
                    "funding_rate",
                    "trade_volume",
                    "trade_count",
                ],
            )
        )

        # Feature selection configuration
        self.feature_selection_config = self.orchestrator_config.get(
            "feature_selection", {}
        )
        self.vif_threshold = float(self.feature_selection_config.get("vif_threshold", 5.0))
        self.mutual_info_threshold = float(
            self.feature_selection_config.get("mutual_info_threshold", 0.01)
        )
        self.lightgbm_importance_threshold = float(
            self.feature_selection_config.get("lightgbm_importance_threshold", 0.01)
        )

        # Multi-timeframe configuration
        self.timeframes = ["1m", "5m", "15m", "30m"]

        # Initialize components (lazy)
        self.triple_barrier_labeler: Any | None = None
        self.advanced_feature_engineer: Any | None = None
        self.autoencoder_generator: Any | None = None
        self.stationarity_checker: VectorizedStationarityChecker | None = None
        self.feature_selector: VectorizedFeatureSelector | None = None
        self.data_normalizer: VectorizedDataNormalizer | None = None

        self.is_initialized = False

        # Debug snapshots for logging
        self._debug_raw_ohlcv: pd.DataFrame | None = None
        self._debug_price_returns: pd.DataFrame | None = None
        self._context_columns: list[str] = []

    def _log_feature_sample(self, stage: str, df: pd.DataFrame, step_no: str) -> None:
        try:
            os.makedirs("log/features_samples", exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_stage = stage.replace(" ", "_")
            fname = f"log/features_samples/{ts}_{step_no}_{safe_stage}.log"
            # Merge raw/returns for visibility when available
            sample = df.copy()
            if self._debug_raw_ohlcv is not None:
                for c in ["open", "high", "low", "close", "volume"]:
                    if c in self._debug_raw_ohlcv.columns and c not in sample.columns:
                        sample[c] = self._debug_raw_ohlcv[c]
            if self._debug_price_returns is not None:
                for c in self._debug_price_returns.columns:
                    if c not in sample.columns:
                        sample[c] = self._debug_price_returns[c]
            # Reorder: raw, returns, then features
            cols_raw = [c for c in ["open", "high", "low", "close", "volume"] if c in sample.columns]
            cols_ret = [c for c in sample.columns if c.endswith("_returns")]
            other = [c for c in sample.columns if c not in cols_raw + cols_ret]
            sample = sample[cols_raw + cols_ret + other]
            with open(fname, "w") as f:
                f.write(f"Stage: {stage} | Step: {step_no} | Shape: {sample.shape}\n")
                f.write(sample.head(50).to_string())
            self.logger.info(f"Feature sample written: {fname}")
        except Exception as e:
            self.logger.warning(f"Failed to write feature sample for {stage}: {e}")

    def _log_feature_errors(self, stage: str, df: pd.DataFrame) -> None:
        try:
            numeric = df.select_dtypes(include=[np.number])
            nan_counts = numeric.isna().sum()
            inf_counts = np.isinf(numeric).sum()
            any_nan = int(nan_counts.sum())
            any_inf = int(inf_counts.sum())
            if any_nan or any_inf:
                self.feature_error_logger.info(
                    f"Stage={stage} | NaN_total={any_nan} | Inf_total={any_inf} | "
                    f"NaN_cols={nan_counts[nan_counts>0].to_dict()} | Inf_cols={inf_counts[inf_counts>0].to_dict()}"
                )
        except Exception:
            pass

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="vectorized labelling orchestrator initialization",
    )
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="vectorized labeling orchestration",
    )
    def _combine_features_and_labels_vectorized(
        self,
        labeled_data: pd.DataFrame,
        advanced_features: dict[str, Any],
    ) -> pd.DataFrame:
        """Combine features and labels using vectorized operations."""
        try:
            # Ensure OHLCV data is present first
            labeled_data = self._ensure_ohlcv_data(labeled_data)

            # Remove metadata columns first
            labeled_data = self._remove_metadata_columns(labeled_data)

            # Remove datetime columns from labeled_data to prevent dtype conflicts
            labeled_data = self._remove_datetime_columns(labeled_data)

            # Attach context columns from stationarity stage if missing
            try:
                _ = self._get_present_context_columns(labeled_data)
                if (
                    self.keep_close_returns
                    and "close_returns" in (self._debug_price_returns.columns if isinstance(self._debug_price_returns, pd.DataFrame) else [])  # noqa: E501
                    and "close_returns" not in labeled_data.columns
                ):
                    labeled_data["close_returns"] = self._debug_price_returns["close_returns"].values  # type: ignore[index]
            except Exception as e:
                self.logger.warning(f"Failed to attach context columns: {e}")

            # If no advanced features, return labeled data as is
            if not advanced_features:
                return labeled_data

            # Build a features DataFrame aligned to labeled_data index
            target_index = labeled_data.index
            num_rows = len(target_index)
            features_df = pd.DataFrame(index=target_index)

            def _as_1d_array(value: Any) -> np.ndarray | None:
                if isinstance(value, pd.Series):
                    return value.values.reshape(-1)
                if isinstance(value, np.ndarray):
                    if value.ndim == 1:
                        return value
                    if value.ndim == 2:
                        if value.shape[0] == 1 or value.shape[1] == 1:
                            return value.reshape(-1)
                        return value[:, 0] if value.shape[1] > 0 else None
                    if value.ndim > 2:
                        try:
                            return value.reshape(-1)
                        except Exception:
                            return value.reshape(value.shape[0], -1)[:, 0] if value.shape[0] > 0 else None  # noqa: E501
                if isinstance(value, list):
                    try:
                        arr = np.asarray(value)
                        if arr.ndim == 1:
                            return arr
                        if arr.ndim == 2:
                            if arr.shape[0] == 1 or arr.shape[1] == 1:
                                return arr.reshape(-1)
                            return arr[:, 0] if arr.shape[1] > 0 else None
                        if arr.ndim > 2:
                            return arr.reshape(-1)
                        return None
                    except Exception:
                        return None
                if isinstance(value, (int, float)):
                    try:
                        if np.isnan(value) or np.isinf(value):
                            return None
                    except Exception:
                        pass
                    # Skip numeric scalars entirely to avoid constant/leaky columns.
                    return None
                if isinstance(value, (str, bool)):
                    return None
                try:
                    if hasattr(value, "__len__") and len(value) > 1:
                        arr2 = np.asarray(value)
                        if arr2.ndim == 1:
                            return arr2
                        if arr2.ndim == 2:
                            return arr2[:, 0] if arr2.shape[1] > 0 else None
                        if arr2.ndim > 2:
                            return arr2.reshape(-1)
                        return None
                except Exception:
                    return None
                return None

            added_columns: list[str] = []
            skipped_scalars: list[str] = []
            scalar_offenders: list[str] = []
            trimmed_aligned: list[str] = []
            padded_aligned: list[str] = []

            for feature_name, feature_value in advanced_features.items():
                arr = _as_1d_array(feature_value)
                if arr is None:
                    skipped_scalars.append(feature_name)
                    scalar_offenders.append(feature_name)
                    if len(skipped_scalars) <= 5:
                        self.logger.debug(
                            f"Skipping feature '{feature_name}': type={type(feature_value)}, "
                            f"shape={getattr(feature_value, 'shape', 'N/A') if hasattr(feature_value, 'shape') else 'N/A'}"
                        )
                    continue

                # Align array length to labeled_data length
                if len(arr) > num_rows:
                    arr = arr[-num_rows:]
                    trimmed_aligned.append(feature_name)
                elif len(arr) < num_rows:
                    pad_size = num_rows - len(arr)
                    arr = np.concatenate([np.full(pad_size, np.nan), arr])
                    padded_aligned.append(feature_name)

                # Add to DataFrame
                try:
                    features_df[feature_name] = pd.to_numeric(arr, errors="coerce")
                    added_columns.append(feature_name)
                except Exception:
                    continue

            if not features_df.empty:
                # Drop columns that are entirely NaN
                features_df = features_df.dropna(axis=1, how="all")

            if not features_df.empty:
                # Remove constant columns
                nunique = features_df.nunique(dropna=True)
                constant_cols = nunique[nunique <= 1].index.tolist()
                if constant_cols:
                    self.logger.warning(
                        f"Dropping {len(constant_cols)} constant features",
                    )
                    preview = constant_cols[:50]
                    self.logger.warning(
                        f"Constant features: {preview}{' ...' if len(constant_cols) > 50 else ''}",
                    )
                    features_df = features_df.drop(columns=constant_cols)

            # Combine with labeled data
            combined_data = pd.concat([labeled_data, features_df], axis=1)
            combined_data = combined_data.loc[:, ~combined_data.columns.duplicated()]

            # Remove raw OHLCV columns to prevent data leakage
            combined_data = self._remove_raw_ohlcv_columns(combined_data)

            # Ensure we also drop any exact baseline raw columns captured at pipeline start
            try:
                baseline_cols: set[str] = {
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "trade_volume",
                    "trade_count",
                    "avg_price",
                    "min_price",
                    "max_price",
                    "funding_rate",
                    "volume_ratio",
                }
                drop_baseline = [
                    c for c in combined_data.columns if c in baseline_cols and c != "label"
                ]
                if drop_baseline:
                    self.logger.warning(
                        f"🚨 Removing baseline/raw columns carried into features: {drop_baseline[:20]}"
                        + (" ..." if len(drop_baseline) > 20 else ""),
                    )
                    combined_data = combined_data.drop(columns=drop_baseline)
            except Exception:
                pass

            try:
                self.logger.info(
                    f"Combined features: added={len(added_columns)}, "
                    f"trimmed={len(trimmed_aligned)}, padded={len(padded_aligned)}, "
                    f"skipped_scalars={len(skipped_scalars)}"
                )
                total_attempted = max(1, len(advanced_features))
                skip_ratio = len(skipped_scalars) / total_attempted
                if skip_ratio > 0.05:
                    self.logger.warning(
                        f"⚠️ Scalar skip ratio ({skip_ratio:.1%}). Some providers may return non-array features; "
                        f"review feature generators. Sample skipped: {skipped_scalars[:10]}",
                    )
                if self.strict_feature_shapes and scalar_offenders:
                    self.logger.warning(
                        f"Strict feature shape check: scalar features detected and skipped: {scalar_offenders[:20]}",
                    )
            except Exception:
                pass

            return combined_data

        except Exception as e:
            try:
                self.logger.exception(
                    f"Error combining features and labels: {e}. "
                    f"labeled_data.shape={labeled_data.shape if isinstance(labeled_data, pd.DataFrame) else 'n/a'}"
                )
            except Exception:
                self.logger.exception(f"Error combining features and labels: {e}")
            return labeled_data

    def _ensure_ohlcv_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Ensure OHLCV data is present in the dataset."""
        required_ohlcv = ["open", "high", "low", "close", "volume"]
        missing_ohlcv = [col for col in required_ohlcv if col not in data.columns]

        if missing_ohlcv:
            self.logger.warning(f"⚠️ Missing OHLCV columns: {missing_ohlcv}")
            if "avg_price" in data.columns:
                if "open" not in data.columns:
                    data["open"] = data["avg_price"]
                if "high" not in data.columns:
                    data["high"] = data["avg_price"]
                if "low" not in data.columns:
                    data["low"] = data["avg_price"]
                if "close" not in data.columns:
                    data["close"] = data["avg_price"]

            if "trade_volume" in data.columns and "volume" not in data.columns:
                data["volume"] = data["trade_volume"]

        return data

    def _remove_metadata_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove metadata columns that are not actual features."""
        metadata_columns = [
            "year",
            "exchange",
            "symbol",
            "timeframe",
            "month",
            "day",
            "day_of_month",
            "quarter",
        ]
        columns_to_remove = [col for col in metadata_columns if col in data.columns]

        if columns_to_remove:
            self.logger.info(f"🗑️ Removing metadata columns: {columns_to_remove}")
            data = data.drop(columns=columns_to_remove)

        return data

    def _remove_datetime_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove datetime columns to prevent dtype conflicts in ML training."""
        try:
            datetime_columns: list[str] = []
            for col in data.columns:
                try:
                    if hasattr(data[col], "dtype") and (
                        data[col].dtype == "datetime64[ns]"
                        or "datetime" in str(data[col].dtype).lower()
                    ):
                        datetime_columns.append(col)
                except (AttributeError, TypeError):
                    continue

            if datetime_columns:
                self.logger.info(f"Removing datetime columns: {datetime_columns}")
                data = data.drop(columns=[c for c in datetime_columns if c in data.columns])

            timestamp_columns = [col for col in data.columns if col.lower() == "timestamp"]
            if timestamp_columns:
                self.logger.info(f"Removing timestamp columns: {timestamp_columns}")
                data = data.drop(columns=timestamp_columns)

            return data

        except Exception as e:
            self.logger.exception(f"Error removing datetime columns: {e}")
            return data

    def _remove_raw_ohlcv_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove raw OHLCV columns to prevent data leakage in ML training."""
        try:
            raw_ohlcv_columns = {
                "open",
                "high",
                "low",
                "close",
                "volume",
                "trade_volume",
                "trade_count",
                "avg_price",
                "min_price",
                "max_price",
                # Treat these as context inputs; engineered variants should be used instead
                "funding_rate",
                "volume_ratio",
                # Exclude raw microstructure proxies; use engineered dynamics instead
                "market_depth",
                "bid_ask_spread",
            }

            ohlcv_columns_found = [col for col in data.columns if col in raw_ohlcv_columns]
            if ohlcv_columns_found:
                self.logger.warning(
                    f"🚨 CRITICAL: Found raw OHLCV columns in features: {ohlcv_columns_found}",
                )
                self.logger.warning(
                    "🚨 Removing raw OHLCV columns to prevent data leakage!",
                )
                data = data.drop(columns=ohlcv_columns_found)

            return data

        except Exception as e:
            self.logger.exception(f"Error removing raw OHLCV columns: {e}")
            return data

    def _remove_stationarity_transform_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove intermediate stationarity helper columns that are not final engineered features."""
        try:
            # Preserve configured context columns (e.g., close_returns and selected volume rep)
            preserve: set[str] = set()
            if self.keep_close_returns and "close_returns" in data.columns:
                preserve.add("close_returns")
            chosen_vol = self._choose_volume_context_column(data)
            if chosen_vol in data.columns:
                preserve.add(chosen_vol)
            to_drop = [
                c
                for c in data.columns
                if (
                    c.endswith(("_log", "_returns", "_log_returns", "_diff", "_detrended"))
                )
                and c not in preserve
            ]
            if to_drop:
                self.logger.info(
                    f"Removing stationarity helper columns: {to_drop[:20]}"
                    + (" ..." if len(to_drop) > 20 else ""),
                )
                data = data.drop(columns=to_drop)
            return data
        except Exception as e:
            self.logger.exception(f"Error removing stationarity helper columns: {e}")
            return data

    def _prepare_final_data_vectorized(
        self, autoencoder_features: pd.DataFrame, labeled_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Prepare final data using vectorized operations."""
        try:
            final_data = pd.concat([autoencoder_features, labeled_data], axis=1)
            final_data = final_data.loc[:, ~final_data.columns.duplicated()]

            if "label" not in final_data.columns and "label" in labeled_data.columns:
                final_data["label"] = labeled_data["label"]

            final_data = final_data.replace([np.inf, -np.inf], np.nan)
            final_data = final_data.fillna(method="ffill").fillna(method="bfill").fillna(0)
            final_data = final_data.dropna(axis=1, how="all")

            # Drop stationarity helper columns (keep engineered features only)
            if "label" in final_data.columns:
                features_only = final_data.drop(columns=["label"])  # type: ignore[call-overload]
                features_only = self._remove_stationarity_transform_columns(features_only)
                final_data = pd.concat([features_only, final_data[["label"]]], axis=1)
            else:
                final_data = self._remove_stationarity_transform_columns(final_data)

            return final_data

        except Exception as e:
            self.logger.exception(f"Error preparing final data: {e}")
            return autoencoder_features

    def _optimize_memory_usage_vectorized(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize memory usage using efficient data types."""
        try:
            optimized_data = data.copy()

            for col in optimized_data.select_dtypes(include=[np.number]).columns:
                col_min = optimized_data[col].min()
                col_max = optimized_data[col].max()
                try:
                    if col_min >= 0 and col_max <= 255:
                        optimized_data[col] = optimized_data[col].astype(np.uint8)
                    elif col_min >= -32768 and col_max <= 32767:
                        optimized_data[col] = optimized_data[col].astype(np.int16)
                    elif col_min >= -2147483648 and col_max <= 2147483647:
                        optimized_data[col] = optimized_data[col].astype(np.int32)
                    else:
                        optimized_data[col] = optimized_data[col].astype(np.float32)
                except Exception:
                    optimized_data[col] = pd.to_numeric(optimized_data[col], errors="coerce").astype(np.float32)  # noqa: E501

            for col in optimized_data.select_dtypes(include=["object"]).columns:
                if optimized_data[col].nunique(dropna=True) < 255:
                    optimized_data[col] = optimized_data[col].astype("category")

            return optimized_data

        except Exception as e:
            self.logger.exception(f"Error optimizing memory usage: {e}")
            return data

    def _save_data_as_parquet(self, data: pd.DataFrame) -> None:
        """Save data as Parquet file."""
        try:
            output_dir = "data/vectorized_features"
            os.makedirs(output_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"vectorized_features_{timestamp}.parquet"
            filepath = os.path.join(output_dir, filename)

            data.to_parquet(filepath, index=True, compression="snappy")
            self.logger.info(f"💾 Data saved as Parquet: {filepath}")
        except Exception as e:
            self.logger.exception(f"Error saving data as Parquet: {e}")

    def _log_dataframe_columns(self, stage: str, df: pd.DataFrame, step_no: str) -> None:
        try:
            os.makedirs("log/features_samples", exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_stage = stage.replace(" ", "_")
            fname = f"log/features_samples/{ts}_{step_no}_{safe_stage}_Columns.txt"
            cols = df.columns.tolist()
            dtypes = df.dtypes.astype(str).to_dict()
            numeric = df.select_dtypes(include=[np.number])
            nan_total = int(numeric.isna().sum().sum()) if not numeric.empty else 0
            inf_total = int(np.isinf(numeric).sum().sum()) if not numeric.empty else 0
            with open(fname, "w") as f:
                f.write(f"Stage: {stage} | Step: {step_no} | Columns: {len(cols)}\n")
                f.write(f"NaN_total={nan_total} | Inf_total={inf_total}\n")
                f.writelines(f"{c}: {dtypes.get(c, 'unknown')}\n" for c in cols)
            self.logger.info(f"Column inventory written: {fname}")
        except Exception as e:
            self.logger.warning(f"Failed to write column inventory for {stage}: {e}")

    def _choose_volume_context_column(self, df: pd.DataFrame) -> str:
        """Local helper to choose volume context column consistent with settings."""
        available = set(df.columns)
        pref = self.volume_representation
        order_map = {
            "returns": [
                "volume_returns",
                "volume_normalized",
                "volume_log",
                "volume_detrended",
                "volume",
            ],
            "normalized": [
                "volume_normalized",
                "volume_returns",
                "volume_log",
                "volume_detrended",
                "volume",
            ],
            "log": [
                "volume_log",
                "volume_returns",
                "volume_normalized",
                "volume_detrended",
                "volume",
            ],
            "detrended": [
                "volume_detrended",
                "volume_returns",
                "volume_normalized",
                "volume_log",
                "volume",
            ],
        }
        for c in order_map.get(pref, []) or []:
            if c in available:
                return c
        for c in [
            "volume_returns",
            "volume_normalized",
            "volume_log",
            "volume_detrended",
            "volume",
        ]:
            if c in available:
                return c
        return "volume"

    def _run_mutual_information_analysis(self, df: pd.DataFrame) -> None:
        """Compute mutual information diagnostics (best-effort)."""
        try:
            from sklearn.feature_selection import (
                mutual_info_classif,
                mutual_info_regression,
            )
            from sklearn.preprocessing import KBinsDiscretizer

            if df is None or df.empty:
                return

            os.makedirs("log/mi", exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")

            meta_label_cols: list[str] = [
                c
                for c in df.columns
                if any(key in c for key in ("STRONG_TREND_CONTINUATION", "RANGE_MEAN_REVERSION", "EXHAUSTION_REVERSAL"))  # noqa: E501
            ]

            exclude_cols = set(self._get_present_context_columns(df)) | {"label"} | set(meta_label_cols)  # noqa: E501
            X_full = df.select_dtypes(include=[np.number]).drop(columns=[c for c in exclude_cols if c in df.columns], errors="ignore")  # noqa: E501
            X_full = X_full.replace([np.inf, -np.inf], np.nan).fillna(0)
            feature_names = X_full.columns.tolist()
            if len(feature_names) == 0:
                self.logger.warning("MI: no numeric features available after exclusions")
                return

            def discretize_features(X: np.ndarray, bins: int) -> np.ndarray:
                try:
                    disc = KBinsDiscretizer(n_bins=bins, encode="ordinal", strategy="quantile")
                    return disc.fit_transform(X)
                except Exception as e:
                    self.logger.warning(
                        f"Failed to discretize features with {bins} bins, using original data. Error: {e}",
                    )
                    return X

            # Classification MI for meta-labels
            classif_reports: dict[str, dict[str, Any]] = {}
            for meta_col in meta_label_cols:
                try:
                    y = df[meta_col].shift(1).fillna(0).astype(int).values
                    if np.unique(y).size < 2:
                        self.logger.info(f"MI(classif): skip {meta_col} (single class)")
                        continue
                    per_bins: dict[str, list[float]] = {}
                    for bins in [5, 10, 20]:
                        Xd = discretize_features(X_full.values, bins)
                        mi = mutual_info_classif(Xd, y, discrete_features=True, random_state=42)
                        per_bins[str(bins)] = [float(v) for v in mi]
                    agg = np.mean(np.vstack([per_bins[b] for b in per_bins]), axis=0)
                    ranking = sorted(zip(feature_names, agg), key=lambda t: t[1], reverse=True)
                    top10 = ranking[:10]
                    self.logger.info(f"MI(classif) {meta_col}: top5={[n for n,_ in top10[:5]]}")
                    classif_reports[meta_col] = {
                        "feature_names": feature_names,
                        "per_bins": per_bins,
                        "agg_mean": [float(v) for v in agg],
                        "top10": [(n, float(s)) for n, s in top10],
                    }
                except Exception as e:
                    self.logger.warning(f"MI(classif) failed for {meta_col}: {e}")

            if classif_reports:
                with open(f"log/mi/{ts}_mi_classif_meta_labels.json", "w") as f:
                    f.write(json.dumps(classif_reports, indent=2))

            # Regression-like MI for label/returns
            regression_reports: dict[str, dict[str, Any]] = {}
            if "label" in df.columns and df["label"].nunique() > 1:
                y_name = "label"
            else:
                y_name = next((c for c in df.columns if c == "close_returns"), None)

            if y_name is not None:
                try:
                    y_series = df[y_name].shift(1).replace([np.inf, -np.inf], np.nan).fillna(0)
                    if y_series.empty or y_series.isna().all():
                        self.logger.warning(f"MI(regress) {y_name}: No valid data available")
                        return
                    y = y_series.values
                    try:
                        is_empty = X_full.empty
                        is_all_nan = X_full.isna().all().all()
                        if is_empty or is_all_nan:
                            self.logger.warning(
                                f"MI(regress) {y_name}: No valid feature data available",
                            )
                            return
                    except Exception as validation_error:
                        self.logger.warning(
                            f"MI(regress) {y_name}: Data validation error: {validation_error}",
                        )
                        return

                    per_bins_r: dict[str, list[float]] = {}
                    for bins in [5, 10, 20]:
                        try:
                            Xd = discretize_features(X_full.values, bins)
                            if Xd.shape[0] != len(y):
                                min_len = min(Xd.shape[0], len(y))
                                Xd = Xd[:min_len]
                                y_aligned = y[:min_len]
                            else:
                                y_aligned = y
                            if Xd.size == 0 or len(y_aligned) == 0:
                                self.logger.warning(
                                    f"MI(regress) {y_name}: Empty data after discretization for bins={bins}"
                                )
                                continue
                            mi_r = mutual_info_regression(Xd, y_aligned, random_state=42)
                            per_bins_r[str(bins)] = [float(v) for v in mi_r]
                        except Exception as bin_error:
                            self.logger.warning(
                                f"MI(regress) {y_name}: Failed for bins={bins}: {bin_error}"
                            )
                            continue

                    if not per_bins_r:
                        self.logger.warning(
                            f"MI(regress) {y_name}: No successful discretization bins",
                        )
                        return

                    try:
                        agg_r = np.mean(np.vstack([per_bins_r[b] for b in per_bins_r]), axis=0)
                        ranking_r = sorted(zip(feature_names, agg_r), key=lambda t: t[1], reverse=True)
                        top10_r = ranking_r[:10]
                        self.logger.info(f"MI(regress) {y_name}: top5={[n for n,_ in top10_r[:5]]}")
                        regression_reports[y_name] = {
                            "feature_names": feature_names,
                            "per_bins": per_bins_r,
                            "agg_mean": [float(v) for v in agg_r],
                            "top10": [(n, float(s)) for n, s in top10_r],
                        }
                    except Exception as agg_error:
                        self.logger.warning(
                            f"MI(regress) {y_name}: Failed to aggregate results: {agg_error}",
                        )
                except Exception as e:
                    self.logger.warning(f"MI(regress) failed for {y_name}: {e}")

            if regression_reports:
                with open(f"log/mi/{ts}_mi_regression.json", "w") as f:
                    f.write(json.dumps(regression_reports, indent=2))

        except Exception as e:
            self.logger.warning(f"MI analysis internal error: {e}")


# -----------------------------------------------------------------------------
# Stationarity Checker
# -----------------------------------------------------------------------------
class VectorizedStationarityChecker:
    """Check and transform data for stationarity using vectorized operations."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("VectorizedStationarityChecker")

    async def check_and_transform_stationarity(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
        order_flow_data: pd.DataFrame | None = None,
    ) -> dict[str, pd.DataFrame | bool]:
        """Check and transform data for stationarity using vectorized operations."""
        try:
            self.logger.info("🔍 Checking data stationarity...")

            price_stationary = self._check_price_stationarity_vectorized(price_data)
            if not price_stationary:
                self.logger.info("📈 Transforming price data for stationarity...")
                price_data = self._transform_price_stationarity_vectorized(price_data)

            volume_stationary = self._check_volume_stationarity_vectorized(volume_data)
            if not volume_stationary:
                self.logger.info("📊 Transforming volume data for stationarity...")
                volume_data = self._transform_volume_stationarity_vectorized(volume_data)

            order_flow_stationary = True
            if order_flow_data is not None:
                order_flow_stationary = self._check_order_flow_stationarity_vectorized(order_flow_data)  # noqa: E501
                if not order_flow_stationary:
                    self.logger.info("🔄 Transforming order flow data for stationarity...")
                    order_flow_data = self._transform_order_flow_stationarity_vectorized(order_flow_data)  # noqa: E501

            return {
                "price_data": price_data,
                "volume_data": volume_data,
                "order_flow_data": order_flow_data,
                "price_stationary": price_stationary,
                "volume_stationary": volume_stationary,
                "order_flow_stationary": order_flow_stationary,
            }

        except Exception as e:
            self.logger.exception(f"Error checking and transforming stationarity: {e}")
            return {
                "price_data": price_data,
                "volume_data": volume_data,
                "order_flow_data": order_flow_data,
                "price_stationary": False,
                "volume_stationary": False,
                "order_flow_stationary": False,
            }

    def _check_price_stationarity_vectorized(self, price_data: pd.DataFrame) -> bool:
        try:
            returns = price_data["close"].pct_change().dropna()
            trend = np.polyfit(range(len(returns)), returns, 1)[0]
            trend_threshold = 0.001
            autocorr = returns.autocorr()
            autocorr_threshold = 0.1
            rolling_std = returns.rolling(20).std()
            variance_ratio = (
                (rolling_std.iloc[-1] / rolling_std.iloc[0]) if len(rolling_std) > 0 else 1.0
            )
            variance_threshold = 2.0
            return (
                abs(trend) < trend_threshold
                and abs(autocorr) < autocorr_threshold
                and variance_ratio < variance_threshold
            )
        except Exception as e:
            self.logger.exception(f"Error checking price stationarity: {e}")
            return False

    def _transform_price_stationarity_vectorized(self, price_data: pd.DataFrame) -> pd.DataFrame:
        try:
            transformed_data = price_data.copy()
            required_columns = ["open", "high", "low", "close", "volume"]
            if not all(col in transformed_data.columns for col in required_columns):
                self.logger.warning(
                    f"Missing required OHLCV columns. Available: {transformed_data.columns.tolist()}",
                )
                return price_data

            # Returns
            transformed_data["close_returns"] = transformed_data["close"].pct_change()
            transformed_data["open_returns"] = transformed_data["open"].pct_change()
            transformed_data["high_returns"] = transformed_data["high"].pct_change()
            transformed_data["low_returns"] = transformed_data["low"].pct_change()

            # Log
            for c in ["close", "open", "high", "low"]:
                transformed_data[f"{c}_log"] = np.log(transformed_data[c].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)  # noqa: E501

            # Detrend
            window = 20
            transformed_data["close_detrended"] = (
                transformed_data["close"] - transformed_data["close"].rolling(window).mean()
            )

            self.logger.info(
                f"✅ Transformed price data shape: {transformed_data.shape}, columns: {transformed_data.columns.tolist()}",
            )
            return transformed_data
        except Exception as e:
            self.logger.exception(f"Error transforming price stationarity: {e}")
            return price_data

    def _check_volume_stationarity_vectorized(self, volume_data: pd.DataFrame) -> bool:
        try:
            if "volume" not in volume_data.columns:
                return True
            volume = volume_data["volume"]
            trend = np.polyfit(range(len(volume)), volume, 1)[0]
            trend_threshold = 0.001
            autocorr = volume.autocorr()
            autocorr_threshold = 0.1
            rolling_std = volume.rolling(20).std()
            variance_ratio = (
                (rolling_std.iloc[-1] / rolling_std.iloc[0]) if len(rolling_std) > 0 else 1.0
            )
            variance_threshold = 2.0
            return (
                abs(trend) < trend_threshold
                and abs(autocorr) < autocorr_threshold
                and variance_ratio < variance_threshold
            )
        except Exception as e:
            self.logger.exception(f"Error checking volume stationarity: {e}")
            return False

    def _transform_volume_stationarity_vectorized(self, volume_data: pd.DataFrame) -> pd.DataFrame:
        try:
            transformed_data = volume_data.copy()
            if "volume" not in transformed_data.columns:
                self.logger.warning(
                    f"Missing volume column. Available: {transformed_data.columns.tolist()}",
                )
                return volume_data

            with np.errstate(divide="ignore", invalid="ignore"):
                transformed_data["volume_returns"] = transformed_data["volume"].pct_change()

            transformed_data["volume_log"] = np.log(transformed_data["volume"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)  # noqa: E501

            window = 20
            transformed_data["volume_detrended"] = (
                transformed_data["volume"] - transformed_data["volume"].rolling(window).mean()
            )
            transformed_data["volume_normalized"] = (
                transformed_data["volume"] / transformed_data["volume"].rolling(window).mean()
            )

            self.logger.info(
                f"✅ Transformed volume data shape: {transformed_data.shape}, columns: {transformed_data.columns.tolist()}",
            )
            return transformed_data
        except Exception as e:
            self.logger.exception(f"Error transforming volume stationarity: {e}")
            return volume_data

    def _check_order_flow_stationarity_vectorized(self, order_flow_data: pd.DataFrame) -> bool:
        try:
            numeric_cols = order_flow_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return True
            first_col = numeric_cols[0]
            data = order_flow_data[first_col]
            trend = np.polyfit(range(len(data)), data, 1)[0]
            trend_threshold = 0.001
            autocorr = data.autocorr()
            autocorr_threshold = 0.1
            return abs(trend) < trend_threshold and abs(autocorr) < autocorr_threshold
        except Exception as e:
            self.logger.exception(f"Error checking order flow stationarity: {e}")
            return False

    def _transform_order_flow_stationarity_vectorized(self, order_flow_data: pd.DataFrame) -> pd.DataFrame:
        try:
            transformed_data = order_flow_data.copy()
            numeric_cols = transformed_data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if (transformed_data[col] > 0).all():
                    transformed_data[f"{col}_log"] = np.log(transformed_data[col])
                window = 20
                transformed_data[f"{col}_detrended"] = (
                    transformed_data[col] - transformed_data[col].rolling(window).mean()
                )
                transformed_data[f"{col}_normalized"] = (
                    transformed_data[col] / transformed_data[col].rolling(window).mean()
                )
            return transformed_data
        except Exception as e:
            self.logger.exception(f"Error transforming order flow stationarity: {e}")
            return order_flow_data


# -----------------------------------------------------------------------------
# Feature Selector
# -----------------------------------------------------------------------------
class VectorizedFeatureSelector:
    """Vectorized feature selector using multiple selection methods."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("VectorizedFeatureSelector")

        self.feature_selection_config = config.get("feature_selection", {})
        self.vif_threshold = float(self.feature_selection_config.get("vif_threshold", 5.0))
        self.mutual_info_threshold = float(
            self.feature_selection_config.get("mutual_info_threshold", 0.001)
        )
        self.lightgbm_importance_threshold = float(
            self.feature_selection_config.get("lightgbm_importance_threshold", 0.001)
        )
        self.min_features_to_keep = int(
            self.feature_selection_config.get("min_features_to_keep", 2)
        )
        self.correlation_threshold = float(
            self.feature_selection_config.get("correlation_threshold", 0.98)
        )
        self.max_removal_percentage = float(
            self.feature_selection_config.get("max_removal_percentage", 0.1)
        )
        self.small_removal_allowance = float(
            self.feature_selection_config.get("small_removal_allowance", 0.05)
        )

        self.enable_constant_removal = bool(
            self.feature_selection_config.get("enable_constant_removal", True)
        )
        self.enable_correlation_removal = bool(
            self.feature_selection_config.get("enable_correlation_removal", True)
        )
        self.enable_vif_removal = bool(
            self.feature_selection_config.get("enable_vif_removal", True)
        )
        self.enable_mutual_info_removal = bool(
            self.feature_selection_config.get("enable_mutual_info_removal", True)
        )
        self.enable_importance_removal = bool(
            self.feature_selection_config.get("enable_importance_removal", True)
        )

        self.enable_safety_checks = bool(
            self.feature_selection_config.get("enable_safety_checks", True)
        )
        self.return_original_on_failure = bool(
            self.feature_selection_config.get("return_original_on_failure", True)
        )

    def _remove_datetime_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            datetime_columns: list[str] = []
            for col in data.columns:
                try:
                    if hasattr(data[col], "dtype") and (
                        data[col].dtype == "datetime64[ns]"
                        or "datetime" in str(data[col].dtype).lower()
                    ):
                        datetime_columns.append(col)
                except (AttributeError, TypeError):
                    continue

            if datetime_columns:
                self.logger.info(f"Removing datetime columns: {datetime_columns}")
                data = data.drop(columns=[c for c in datetime_columns if c in data.columns])

            timestamp_columns = [col for col in data.columns if col.lower() == "timestamp"]
            if timestamp_columns:
                self.logger.info(f"Removing timestamp columns: {timestamp_columns}")
                data = data.drop(columns=timestamp_columns)

            return data

        except Exception as e:
            self.logger.exception(f"Error removing datetime columns: {e}")
            return data

    async def select_optimal_features(
        self, data: pd.DataFrame, labels: np.ndarray | None = None
    ) -> pd.DataFrame:
        try:
            self.logger.info("🎯 Starting feature selection...")
            original_data = data.copy()

            data = self._remove_datetime_columns(data)

            self.logger.info(
                f"🔍 Analyzing {len(data.columns)} features before selection...",
            )
            feature_analysis: dict[str, dict[str, Any]] = {}
            for col in data.columns:
                if col in data.columns:
                    feature_analysis[col] = {
                        "dtype": str(data[col].dtype),
                        "nunique": int(data[col].nunique()),
                        "has_nan": bool(data[col].isnull().any()),
                        "nan_count": int(data[col].isnull().sum()),
                        "min": data[col].min() if data[col].dtype in ["float64", "float32", "int64", "int32"] else None,  # noqa: E501
                        "max": data[col].max() if data[col].dtype in ["float64", "float32", "int64", "int32"] else None,  # noqa: E501
                        "mean": data[col].mean() if data[col].dtype in ["float64", "float32", "int64", "int32"] else None,  # noqa: E501
                        "std": data[col].std() if data[col].dtype in ["float64", "float32", "int64", "int32"] else None,  # noqa: E501
                    }

            constant_features = [col for col, info in feature_analysis.items() if info["nunique"] <= 1]
            low_variance_features = [
                col
                for col, info in feature_analysis.items()
                if info["dtype"] in ["float64", "float32", "int64", "int32"]
                and info["std"] is not None
                and float(info["std"]) < 1e-6
            ]

            self.logger.info(
                f"📊 Feature analysis: {len(constant_features)} constant, {len(low_variance_features)} low-variance features",
            )

            self.logger.info("Handling NaN values in feature selection...")
            nan_counts = data.isnull().sum()
            nan_features = nan_counts[nan_counts > 0]
            if len(nan_features) > 0:
                self.logger.info(
                    f"📊 Found NaN values in {len(nan_features)} features:",
                )
                for feature, count in nan_features.items():
                    percentage = (int(count) / max(1, len(data))) * 100
                    self.logger.info(
                        f"   {feature}: {int(count):,} NaN values ({percentage:.2f}%)",
                    )
            else:
                self.logger.info("✅ No NaN values found in any features")

            data = data.fillna(method="ffill").fillna(method="bfill").fillna(0)

            if self.enable_constant_removal:
                constant_to_drop: list[str] = []
                for col in data.columns:
                    if data[col].nunique() <= 1:
                        constant_to_drop.append(col)
                    elif data[col].dtype in ["float64", "float32", "int64", "int32"]:
                        variance = float(data[col].var())
                        if variance < 1e-10:
                            constant_to_drop.append(col)
                if constant_to_drop:
                    self.logger.info(
                        f"Removed {len(constant_to_drop)} constant/low-variance features: {constant_to_drop[:20]}"
                        + (" ..." if len(constant_to_drop) > 20 else ""),
                    )
                    data = data.drop(columns=constant_to_drop)

            if self.enable_safety_checks and len(data.columns) < self.min_features_to_keep:
                self.logger.warning(
                    f"Too few features after constant removal ({len(data.columns)}). Skipping further selection.",
                )
                return data

            if (
                self.enable_correlation_removal
                and len(data.columns) > self.min_features_to_keep
                and len(data.columns) > 2
            ):
                correlated_features = self._remove_correlated_features_vectorized(data)
                if correlated_features:
                    removal_percentage = len(correlated_features) / max(1, len(data.columns))
                    should_remove = False
                    if removal_percentage <= self.max_removal_percentage:
                        should_remove = True
                    elif removal_percentage <= self.small_removal_allowance:
                        if len(data.columns) - len(correlated_features) >= self.min_features_to_keep:
                            should_remove = True
                            self.logger.info(
                                f"✅ Allowing small correlated feature removal ({removal_percentage:.2f}%) despite being above threshold",
                            )
                    if should_remove and len(data.columns) - len(correlated_features) >= self.min_features_to_keep:
                        self.logger.info(
                            f"Removed {len(correlated_features)} highly correlated features",
                        )
                        with contextlib.suppress(Exception):
                            self.logger.info(
                                f"Correlated features removed: {sorted(correlated_features)[:20]}"
                                + (" ..." if len(correlated_features) > 20 else ""),
                            )
                        data = data.drop(columns=correlated_features)
                    else:
                        self.logger.info(
                            f"Skipping correlated feature removal (removal %: {removal_percentage:.2f} > threshold: {self.max_removal_percentage:.2f})",
                        )

            self.logger.info(
                f"📊 Features after correlated feature removal: {len(data.columns)}",
            )

            if (
                self.enable_mutual_info_removal
                and labels is not None
                and len(labels) > 0
                and len(data.columns) > self.min_features_to_keep
            ):
                all_low_mi_features = self._remove_low_mutual_info_features_vectorized(data, labels)
                if all_low_mi_features:
                    max_features_to_remove = int(len(data.columns) * self.max_removal_percentage)
                    low_mi_features = all_low_mi_features[:max_features_to_remove]
                    if len(data.columns) - len(low_mi_features) >= self.min_features_to_keep:
                        self.logger.info(
                            f"Removed {len(low_mi_features)} low mutual information features (limited to {len(low_mi_features) / max(1, len(data.columns)):.2f}% of total features)",
                        )
                        with contextlib.suppress(Exception):
                            self.logger.info(
                                f"Low MI features removed: {sorted(low_mi_features)[:20]}"
                                + (" ..." if len(low_mi_features) > 20 else ""),
                            )
                        data = data.drop(columns=low_mi_features)
                    else:
                        self.logger.info(
                            f"Skipping mutual info feature removal (would leave too few features: {len(data.columns) - len(low_mi_features)} < {self.min_features_to_keep})",
                        )
                else:
                    self.logger.info("No low mutual information features found to remove")

            self.logger.info(
                f"📊 Features after mutual information removal: {len(data.columns)}",
            )

            if (
                self.enable_importance_removal
                and labels is not None
                and len(labels) > 0
                and len(data.columns) > self.min_features_to_keep
            ):
                all_low_importance_features = self._remove_low_importance_features_vectorized(data, labels)
                if all_low_importance_features:
                    max_features_to_remove = int(len(data.columns) * self.max_removal_percentage)
                    low_importance_features = all_low_importance_features[:max_features_to_remove]
                    if len(data.columns) - len(low_importance_features) >= self.min_features_to_keep:
                        self.logger.info(
                            f"Removed {len(low_importance_features)} low importance features (limited to {len(low_importance_features) / max(1, len(data.columns)):.2f}% of total features)",
                        )
                        with contextlib.suppress(Exception):
                            self.logger.info(
                                f"Low-importance features removed: {sorted(low_importance_features)[:20]}"
                                + (" ..." if len(low_importance_features) > 20 else ""),
                            )
                        data = data.drop(columns=low_importance_features)
                    else:
                        self.logger.info(
                            f"Skipping importance feature removal (would leave too few features: {len(data.columns) - len(low_importance_features)} < {self.min_features_to_keep})",
                        )
                else:
                    self.logger.info("No low importance features found to remove")

            final_features = len(data.columns)
            self.logger.info(
                f"Feature selection completed. Initial features: {original_data.shape[1]}, Final features: {final_features}",
            )

            if final_features == 0:
                self.logger.error("No features remaining after selection!")
                if self.return_original_on_failure:
                    self.logger.info("Returning original data as fallback.")
                    return self._remove_datetime_columns(original_data)
                raise ValueError("No features remaining after selection and fallback disabled")

            return data

        except Exception as e:
            self.logger.exception(f"Error in feature selection: {e}")
            if self.return_original_on_failure:
                self.logger.info("Returning original data as fallback due to error.")
                return self._remove_datetime_columns(original_data)
            raise

    def _remove_correlated_features_vectorized(self, data: pd.DataFrame) -> list[str]:
        try:
            from sklearn.impute import SimpleImputer

            imputer = SimpleImputer(strategy="median")
            data_imputed = pd.DataFrame(
                imputer.fit_transform(data), columns=data.columns, index=data.index
            )

            correlation_matrix = data_imputed.corr()
            upper_triangle = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            high_correlation = (np.abs(correlation_matrix) > self.correlation_threshold) & upper_triangle

            to_drop: list[str] = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i + 1, len(correlation_matrix.columns)):
                    if bool(high_correlation.iloc[i, j]):
                        to_drop.append(correlation_matrix.columns[j])

            return list(set(to_drop))
        except Exception as e:
            self.logger.exception(f"Error removing correlated features: {e}")
            return []

    def _remove_low_mutual_info_features_vectorized(
        self, data: pd.DataFrame, labels: np.ndarray, max_removal_percentage: float | None = None
    ) -> list[str]:
        try:
            from sklearn.feature_selection import mutual_info_classif
            from sklearn.impute import SimpleImputer

            imputer = SimpleImputer(strategy="median")
            data_imputed = pd.DataFrame(
                imputer.fit_transform(data), columns=data.columns, index=data.index
            )

            mi_scores = mutual_info_classif(data_imputed, labels, random_state=42)
            feature_importance: dict[str, float] = dict(zip(data_imputed.columns, mi_scores))
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1])
            low_mi_features = [col for col, score in feature_importance.items() if score < self.mutual_info_threshold]  # noqa: E501

            if max_removal_percentage is not None:
                max_features_to_remove = int(len(data.columns) * max_removal_percentage)
                return [col for col, _ in sorted_features[:max_features_to_remove]]

            return low_mi_features
        except Exception as e:
            self.logger.exception(f"Error removing low mutual information features: {e}")
            return []

    def _remove_low_importance_features_vectorized(
        self, data: pd.DataFrame, labels: np.ndarray, max_removal_percentage: float | None = None
    ) -> list[str]:
        try:
            import lightgbm as lgb
            from sklearn.impute import SimpleImputer

            imputer = SimpleImputer(strategy="median")
            data_imputed = pd.DataFrame(
                imputer.fit_transform(data), columns=data.columns, index=data.index
            )

            model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42,
                verbose=-1,
            )
            import warnings as _warnings

            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                model.fit(data_imputed, labels)

            feature_importance: dict[str, float] = dict(
                zip(data_imputed.columns, model.feature_importances_)
            )
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1])
            low_importance_features = [
                col for col, importance in feature_importance.items()
                if float(importance) < self.lightgbm_importance_threshold
            ]

            if max_removal_percentage is not None:
                max_features_to_remove = int(len(data.columns) * max_removal_percentage)
                return [col for col, _ in sorted_features[:max_features_to_remove]]

            return low_importance_features
        except Exception as e:
            self.logger.exception(f"Error removing low importance features: {e}")
            return []


# -----------------------------------------------------------------------------
# Data Normalizer
# -----------------------------------------------------------------------------
class VectorizedDataNormalizer:
    """Normalize data using various scaling methods with vectorized operations."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("VectorizedDataNormalizer")

        self.normalization_config = config.get("data_normalization", {})
        self.scaling_method = self.normalization_config.get("scaling_method", "robust")
        self.outlier_handling = self.normalization_config.get("outlier_handling", "clip")
        orch_cfg = config.get("vectorized_labelling_orchestrator", {})
        self.keep_close_returns = orch_cfg.get("keep_only_close_returns_main", True)
        self.volume_representation = str(orch_cfg.get("volume_representation", "returns")).lower()
        self.context_non_feature_columns: set[str] = set(
            orch_cfg.get(
                "context_non_feature_columns",
                [
                    "year",
                    "month",
                    "day",
                    "exchange",
                    "symbol",
                    "timeframe",
                    "timestamp",
                ],
            )
        )

    async def normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            self.logger.info("📏 Normalizing data...")
            data = self._remove_datetime_columns(data)
            data = self._clip_outliers_vectorized(data)
            data = self._apply_robust_scaling_vectorized(data)
            self.logger.info("✅ Data normalization completed")
            return data
        except Exception as e:
            self.logger.exception(f"Error normalizing data: {e}")
            return data

    def _remove_datetime_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            datetime_columns: list[str] = []
            for col in data.columns:
                try:
                    if hasattr(data[col], "dtype") and (
                        data[col].dtype == "datetime64[ns]"
                        or "datetime" in str(data[col].dtype).lower()
                    ):
                        datetime_columns.append(col)
                except (AttributeError, TypeError):
                    continue

            if datetime_columns:
                self.logger.info(f"Removing datetime columns: {datetime_columns}")
                data = data.drop(columns=[c for c in datetime_columns if c in data.columns])

            timestamp_columns = [col for col in data.columns if col.lower() == "timestamp"]
            if timestamp_columns:
                self.logger.info(f"Removing timestamp columns: {timestamp_columns}")
                data = data.drop(columns=timestamp_columns)

            return data

        except Exception as e:
            self.logger.exception(f"Error removing datetime columns: {e}")
            return data

    def _clip_outliers_vectorized(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            clipped_data = data.copy()
            for col in clipped_data.select_dtypes(include=[np.number]).columns:
                try:
                    Q1 = clipped_data[col].quantile(0.25)
                    Q3 = clipped_data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    clipped_data[col] = np.clip(clipped_data[col].values, lower_bound, upper_bound)
                except Exception as col_error:
                    self.logger.warning(f"Error clipping column {col}: {col_error}")
                    continue
            return clipped_data
        except Exception as e:
            self.logger.exception(f"Error clipping outliers: {e}")
            return data

    def _apply_robust_scaling_vectorized(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            from sklearn.preprocessing import RobustScaler

            scaler = RobustScaler()
            numeric_cols = list(data.select_dtypes(include=[np.number]).columns)
            exclude: set[str] = {"label"}
            if self.keep_close_returns and "close_returns" in data.columns:
                exclude.add("close_returns")
            chosen_vol_col = self._choose_volume_context_column(data)
            if chosen_vol_col in data.columns:
                exclude.add(chosen_vol_col)
            exclude |= {c for c in self.context_non_feature_columns if c in data.columns}
            scale_cols = [c for c in numeric_cols if c not in exclude]
            if scale_cols:
                data[scale_cols] = scaler.fit_transform(data[scale_cols])
            return data
        except Exception as e:
            self.logger.exception(f"Error applying robust scaling: {e}")
            return data

    def _choose_volume_context_column(self, df: pd.DataFrame) -> str:
        available = set(df.columns)
        pref = self.volume_representation
        order_map = {
            "returns": [
                "volume_returns",
                "volume_normalized",
                "volume_log",
                "volume_detrended",
                "volume",
            ],
            "normalized": [
                "volume_normalized",
                "volume_returns",
                "volume_log",
                "volume_detrended",
                "volume",
            ],
            "log": [
                "volume_log",
                "volume_returns",
                "volume_normalized",
                "volume_detrended",
                "volume",
            ],
            "detrended": [
                "volume_detrended",
                "volume_returns",
                "volume_normalized",
                "volume_log",
                "volume",
            ],
        }
        for c in order_map.get(pref, []) or []:
            if c in available:
                return c
        for c in [
            "volume_returns",
            "volume_normalized",
            "volume_log",
            "volume_detrended",
            "volume",
        ]:
            if c in available:
                return c
        return "volume"
