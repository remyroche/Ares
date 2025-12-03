#!/usr/bin/env python3
"""Specialist Feature Diagnostics CLI.

Analyzes specialist model outputs (Risk, Liquidity, Breakout/Bounce)
loaded via `get_specialist_models_outputs` against meta-labeling targets
produced by `FeatureGenerationMetaLabelingStep`.

Metrics per specialist feature:
- Event-aware correlation-based MI proxy (cheap mutual information proxy)
- MI stability across time-series CV folds (mean and coefficient of variation)
- Pearson correlation with target label and corresponding R^2

Outputs Markdown and CSV reports under the `outcomes/` directory.
Optionally restricts analysis to the last N calendar days via --lookback-days.

Usage example (from project root):

  # Uses direction-aware default target (binary_label_long for longs)
  python scripts/specialist_feature_diagnostics.py \
      --symbol ETHUSDT --exchange binance --timeframe 15m \
      --direction long --regime-timeframe 1h --lookback-days 365

  # Explicit target column (regression target for regressors)
  python scripts/specialist_feature_diagnostics.py \
      --symbol ETHUSDT --exchange binance --timeframe 15m \
      --direction long --target-col target_long --regime-timeframe 1h

  # Compare multiple targets (classifiers vs regressors)
  python scripts/specialist_feature_diagnostics.py \
      --symbol ETHUSDT --exchange binance --timeframe 15m \
      --direction long --compare-targets --regime-timeframe 1h
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import system_logger  # type: ignore
from src.utils.ml_common.get_specialist_models_outputs import (  # type: ignore
    get_specialist_models_outputs,
)
from src.training.steps.labeling.feature_generation_meta_labeling_step import (  # type: ignore
    FeatureGenerationMetaLabelingStep,
)
from src.training.steps.labeling.snr_diagnostics import (  # type: ignore
    _load_labeled_data,
)
from src.training.steps.pre_training.components.final_feature_selection import (  # type: ignore
    FinalFeatureSelectionConfig,
    FinalFeatureSelectionComponent,
)
from src.training.steps.market_analysis import step_registry  # type: ignore


logger = system_logger.getChild("specialist_feature_diagnostics")

OUTCOMES_DIR = Path("outcomes")


async def _run_specialist_training(
    symbol: str,
    exchange: str,
    timeframe: str,
    direction: str,
    regime_timeframe: str,
    lookback_days: Optional[float] = None,
    selected_specialists: Optional[list[str]] = None,
) -> None:
    """Run training for all (or selected) specialist models sequentially."""
    logger.info("🚀 Starting specialist model training sequence")

    # Order matters: some steps might depend on artifacts from others,
    # though ideally they should be independent or use shared feature generation.
    default_specialist_steps = [
        "ml_volume_force_step",
        "ml_breakout_bounce_regime_step",
        "ml_mean_reversion_step",
        "xgb_meso_regime",
        "hmm_macro_regime",
        "ml_smc_regime_step",
        "ml_liquidity_regime_step",
        "ml_risk_regime_step",
        "ml_path_regime_step",
    ]

    specialist_steps = selected_specialists if selected_specialists else default_specialist_steps

    config_base = {
        "symbol": symbol,
        "exchange": exchange,
        "timeframe": timeframe,
        "direction": direction,
        "regime_timeframe": regime_timeframe,
        "execution_mode": "full", # default to full for training
    }

    if lookback_days:
        config_base["lookback_days"] = lookback_days

    for step_key in specialist_steps:
        # Verify registry key and handle potential aliases (e.g. missing _step suffix)
        try:
             if not step_registry.is_registered(step_key):
                 if step_registry.is_registered(step_key + "_step"):
                     step_key = step_key + "_step"
                 else:
                     logger.warning(f"⚠️ Step '{step_key}' not found in registry. Skipping.")
                     continue
        except Exception:
             logger.warning(f"⚠️ Error checking registry for '{step_key}'. Skipping.")
             continue

        logger.info(f"▶️ Running {step_key}...")
        try:
            StepClass = step_registry.get_step(step_key)
            step_instance = StepClass(step_name=step_key)

            # Run the step
            result = await step_instance.run(config_base)

            if result.get("success"):
                logger.info(f"✅ {step_key} completed successfully.")
            else:
                logger.error(f"❌ {step_key} failed: {result.get('error')}")
        except Exception as exc:
            logger.error(f"❌ Exception running {step_key}: {exc}")

    logger.info("🏁 Specialist model training sequence finished.")


def _ensure_outcomes_dir() -> Path:
    """Ensure outcomes directory exists and return it."""
    OUTCOMES_DIR.mkdir(exist_ok=True)
    return OUTCOMES_DIR


def _export_report(
    prefix: str,
    symbol: str,
    exchange: str,
    timeframe: str,
    direction: str,
    model: str,
    payload: Dict[str, Any],
    markdown_lines: list[str],
) -> Tuple[Path, Path]:
    """Export diagnostics payload as Markdown and CSV into outcomes/.

    Filenames are of the form:
        outcomes/{prefix}_{symbol}_{timeframe}_{YYYYMMDD_%H%M%S}.md/csv
    """
    out_dir = _ensure_outcomes_dir()
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base_name = f"{prefix}_{symbol}_{timeframe}_{ts}"

    md_path = out_dir / f"{base_name}.md"
    csv_path = out_dir / f"{base_name}.csv"

    with md_path.open("w") as f_md:
        f_md.write("\n".join(markdown_lines))

    # Optional CSV export when feature_metrics are provided in the payload
    feature_metrics = payload.get("feature_metrics")
    model_reliability = payload.get("model_reliability")
    model_pairwise = payload.get("model_pairwise")

    if isinstance(feature_metrics, dict) and feature_metrics:
        try:
            frames: list[pd.DataFrame] = []

            # Base per-feature metrics
            df_features = pd.DataFrame.from_dict(feature_metrics, orient="index")
            df_features.index.name = "feature"
            df_features.insert(0, "row_type", "feature")
            frames.append(df_features)

            # Optional per-model reliability metrics
            if isinstance(model_reliability, dict):
                per_model = model_reliability.get("per_model")
                if isinstance(per_model, dict) and per_model:
                    df_models = pd.DataFrame.from_dict(per_model, orient="index")
                    df_models.index.name = "feature"
                    df_models.insert(0, "row_type", "model_reliability")
                    frames.append(df_models)

            # Optional pairwise relationships between specialist models
            if isinstance(model_pairwise, dict):
                pairs = model_pairwise.get("pairs")
                if isinstance(pairs, list) and pairs:
                    df_pairs = pd.DataFrame(pairs)
                    if not df_pairs.empty:
                        df_pairs = df_pairs.copy()
                        pair_index = (
                            df_pairs["model_i"].astype(str)
                            + "|"
                            + df_pairs["model_j"].astype(str)
                        )
                        df_pairs.index = pair_index
                        df_pairs.index.name = "feature"
                        df_pairs.insert(0, "row_type", "model_pairwise")
                        frames.append(df_pairs)

            if frames:
                df_all = pd.concat(frames, axis=0, sort=False)
                df_all.to_csv(csv_path)
                logger.info(
                    "Saved %s diagnostics to %s and %s",
                    prefix,
                    md_path,
                    csv_path,
                )
            else:
                # Fallback: feature metrics only (should not normally trigger)
                df = pd.DataFrame.from_dict(feature_metrics, orient="index")
                df.index.name = "feature"
                df.to_csv(csv_path)
                logger.info(
                    "Saved %s diagnostics to %s and %s",
                    prefix,
                    md_path,
                    csv_path,
                )
        except Exception as csv_exc:  # pragma: no cover - best-effort CSV export
            logger.warning("Failed to export CSV diagnostics table: %s", csv_exc)
    else:
        logger.info("Saved %s diagnostics to %s", prefix, md_path)

    return md_path, csv_path


def _prepare_labels(
    symbol: str,
    exchange: str,
    timeframe: str,
    direction: str,
    model: str,
    target_col: str,
    lookback_days: Optional[float] = None,
) -> Tuple[pd.Series, pd.DatetimeIndex, pd.Series]:
    """Load labeled_data and return (y, datetime index, realized_return).

    Uses the same loader as snr_diagnostics to ensure compatibility with
    FeatureGenerationMetaLabelingStep artifacts.
    
    Returns:
        Tuple of (target_series, training_index, realized_return_series)
    """
    labeled_df = _load_labeled_data(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        direction=direction,
        model=model,
    )

    if target_col not in labeled_df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in labeled_data; "
            f"available columns: {sorted(labeled_df.columns)}"
        )

    # Normalize timestamp index. Prefer an existing DatetimeIndex on
    # labeled_df if available; only fall back to timestamp/close_time
    # columns when the index is not already datetime-like.
    if isinstance(labeled_df.index, pd.DatetimeIndex):
        idx = labeled_df.index
        if idx.tz is not None:
            try:
                idx = idx.tz_convert("UTC").tz_localize(None)
            except Exception:
                idx = idx.tz_localize(None)
        labeled_df = labeled_df.copy()
        labeled_df.index = idx
    elif "timestamp" in labeled_df.columns:
        ts_raw = labeled_df["timestamp"]

        # Handle both datetime-like and numeric Unix timestamp columns.
        # Meta-labeling typically stores exchange timestamps in milliseconds,
        # so we interpret numeric values as ms since epoch to avoid bogus
        # 1970-01-01 ranges.
        if pd.api.types.is_datetime64_any_dtype(ts_raw):
            ts = pd.to_datetime(ts_raw, utc=True, errors="coerce")
        else:
            numeric = pd.to_numeric(ts_raw, errors="coerce")
            if numeric.notna().any():
                ts = pd.to_datetime(numeric, unit="ms", utc=True, errors="coerce")
            else:
                ts = pd.to_datetime(ts_raw, utc=True, errors="coerce")

        try:
            ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
        except Exception:
            ts = ts.dt.tz_localize(None)

        valid_mask = ~ts.isna()
        labeled_df = labeled_df.loc[valid_mask].copy()
        ts = ts[valid_mask]
        labeled_df.index = ts
    elif "close_time" in labeled_df.columns:
        # Fallback for labeled_data artifacts that store event timestamps
        # in a 'close_time' column rather than a generic 'timestamp'.
        close_col = labeled_df["close_time"]
        try:
            if pd.api.types.is_datetime64_any_dtype(close_col):
                ts = pd.to_datetime(close_col, utc=True, errors="coerce")
            else:
                # Infer epoch unit (s, ms, ns) from numeric magnitude to avoid
                # misinterpreting seconds as milliseconds (which would produce
                # bogus 1970-era timestamps).
                close_numeric = pd.to_numeric(close_col, errors="coerce")
                if close_numeric.notna().any():
                    q = float(close_numeric.dropna().quantile(0.5))
                    if q > 1e14:
                        unit = "ns"
                    elif q > 1e11:
                        unit = "ms"
                    elif q > 1e9:
                        unit = "s"
                    else:
                        unit = "ms"
                    ts = pd.to_datetime(close_numeric, unit=unit, utc=True, errors="coerce")
                else:
                    ts = pd.to_datetime(close_col, utc=True, errors="coerce")
        except Exception:
            ts = pd.to_datetime(close_col, utc=True, errors="coerce")
        try:
            ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
        except Exception:
            ts = ts.dt.tz_localize(None)
        valid_mask = ~ts.isna()
        labeled_df = labeled_df.loc[valid_mask].copy()
        ts = ts[valid_mask]
        labeled_df.index = ts
    else:
        raise ValueError(
            "labeled_data has neither 'timestamp'/'close_time' column nor DatetimeIndex"
        )

    y = labeled_df[target_col].astype(float)
    valid_y = ~y.isna()
    y = y[valid_y]

    # Optional time-based lookback restriction
    if lookback_days is not None and lookback_days > 0:
        try:
            cutoff = y.index.max() - pd.Timedelta(days=float(lookback_days))
            y = y.loc[y.index >= cutoff]
        except Exception as lb_exc:
            logger.warning("Failed to apply lookback_days filter: %s", lb_exc)

    # Debug: inspect target distribution after cleaning/lookback
    try:
        vc = y.value_counts(dropna=False).to_dict()
        logger.info(
            "🎯 Target '%s' distribution after cleaning/lookback_days=%s: n=%d, counts=%s",
            target_col,
            str(lookback_days),
            len(y),
            vc,
        )
    except Exception as dist_exc:
        logger.warning("Failed to compute target distribution summary: %s", dist_exc)

    if len(y) < 100:
        logger.warning(
            "Only %d valid target samples after cleaning/lookback; diagnostics may be noisy",
            len(y),
        )

    # Diagnostics: report training index range used for specialist alignment
    if isinstance(y.index, pd.DatetimeIndex) and len(y.index) > 0:
        logger.info(
            "🎯 Labeled training index range (%s): %s → %s (n=%d)",
            target_col,
            y.index.min(),
            y.index.max(),
            len(y.index),
        )

    # Extract realized_return for PnL simulation (if available)
    if "realized_return" in labeled_df.columns:
        realized_return = labeled_df["realized_return"].astype(float)
        realized_return = realized_return.loc[y.index]
    else:
        # Fallback: use the target column as a proxy if it's a continuous value
        realized_return = pd.Series(index=y.index, dtype=float)
        logger.warning("realized_return column not found in labeled_data; PnL simulation will be limited")

    return y, y.index, realized_return


def _load_specialist_features(
    symbol: str,
    exchange: str,
    base_timeframe: str,
    regime_timeframe: str,
    direction: str,
    model: str,
    training_index: pd.DatetimeIndex,
    enable_risk_hmm_specialist: bool,
) -> pd.DataFrame:
    """Load specialist model outputs aligned to training_index.

    Uses FeatureGenerationMetaLabelingStep's BaseStep machinery to obtain an
    ArtifactRouter instance and then delegates to get_specialist_models_outputs.
    """
    step = FeatureGenerationMetaLabelingStep()
    step.set_context(
        symbol=symbol,
        exchange=exchange,
        timeframe=base_timeframe,
        direction=direction,
        model=model,
    )

    specialist_config: Dict[str, Any] = {
        "symbol": symbol,
        "exchange": exchange,
        "timeframe": base_timeframe,
        "regime_timeframe": regime_timeframe,
        "direction": direction,
        # Mirror training behavior: include the optional HMM risk specialist
        # only when explicitly enabled so diagnostics can match the exact
        # specialist feature block used for training.
        "enable_risk_hmm_specialist": enable_risk_hmm_specialist,
        # Use the same canonical per-specialist scalar projection as unified
        # training so diagnostics operate on the identical feature block when
        # projection_mode="raw".
        "use_canonical_specialist_scalars": True,
    }

    specialist_df = get_specialist_models_outputs(
        artifact_router=step.artifact_router,
        training_index=training_index,
        config=specialist_config,
        logger=step.logger,
        strict=False,
    )

    if specialist_df is None or specialist_df.empty:
        raise ValueError(
            "No specialist model outputs found for the given context; "
            "ensure the specialist steps have been run first."
        )

    # Ensure strict alignment to the training index
    specialist_df = specialist_df.reindex(training_index, method="ffill")

    # Keep only numeric columns; let downstream metrics logic drop degenerate
    # features (all-NaN or zero variance) rather than failing early here.
    numeric = specialist_df.select_dtypes(include=[np.number, "bool"]).copy()
    if numeric.shape[1] == 0:
        raise ValueError("No numeric specialist features found in specialist_df")

    return numeric


def _compute_feature_metrics(
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int = 5,
) -> Dict[str, Dict[str, float]]:
    """Compute MI proxy, MI stability, correlation, and R^2 per feature."""
    # Align and clean
    common_index = X.index.intersection(y.index)
    X = X.loc[common_index].copy()
    y = y.loc[common_index].astype(float)

    mask = ~y.isna()
    X = X.loc[mask]
    y = y.loc[mask]

    # Replace infinities and fill remaining NaNs in X for robust correlation-based metrics
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Pre-compute numeric target array and detect binary vs continuous target
    y_arr = y.to_numpy(dtype=float)
    uniq = np.unique(y_arr[~np.isnan(y_arr)])
    uniq_rounded = np.unique(np.round(uniq).astype(int)) if uniq.size > 0 else np.array([])
    is_binary_target = bool(uniq_rounded.size > 0 and set(uniq_rounded).issubset({0, 1}))

    # Debug: log alignment statistics and class distribution actually
    # entering the MI / R^2 computation.
    try:
        logger.info(
            "📊 Feature metrics alignment: n_samples=%d, n_features=%d, is_binary=%s, classes=%s",
            len(y),
            X.shape[1],
            is_binary_target,
            uniq_rounded.tolist() if uniq_rounded.size > 0 else [],
        )
        if is_binary_target:
            class_counts = {
                int(c): int((y_arr == c).sum()) for c in uniq_rounded
            }
            logger.info(
                "📊 Binary target class counts after alignment: %s",
                class_counts,
            )
    except Exception as align_exc:
        logger.warning("Failed to log feature-metrics alignment statistics: %s", align_exc)

    if len(y) < max(cv_folds * 5, 50):
        logger.warning(
            "Limited samples (%d) for CV=%d; MI stability estimates may be noisy",
            len(y),
            cv_folds,
        )

    # Use FinalFeatureSelectionComponent utilities for MI proxy and stability by default
    config = FinalFeatureSelectionConfig()
    component = FinalFeatureSelectionComponent(config=config)

    if is_binary_target:
        # Binary-aware MI proxy: use absolute Pearson correlation as a cheap MI proxy
        mi_full_series = {}
        for col in X.columns:
            x_full = X[col].to_numpy(dtype=float)
            try:
                c_full = float(np.corrcoef(x_full, y_arr)[0, 1])
            except Exception:
                c_full = float("nan")
            mi_full_series[col] = abs(c_full) if np.isfinite(c_full) else 0.0
        mi_full = pd.Series(mi_full_series, index=X.columns)

        # MI stability across TimeSeriesSplit folds using the same correlation proxy
        mi_mean: Dict[str, float] = {}
        mi_cv: Dict[str, float] = {}
        tscv = TimeSeriesSplit(n_splits=cv_folds)

        for col in X.columns:
            fold_scores: list[float] = []
            for train_idx, _ in tscv.split(X):
                y_tr = y_arr[train_idx]
                x_tr = X.iloc[train_idx][col].to_numpy(dtype=float)

                # Clean NaNs in y
                mask_tr = ~np.isnan(y_tr)
                if not np.any(mask_tr):
                    continue
                y_tr_clean = y_tr[mask_tr]
                x_tr_clean = x_tr[mask_tr]

                # Require both classes and some variation in the feature
                if len(np.unique(y_tr_clean)) < 2 or np.all(x_tr_clean == x_tr_clean[0]):
                    continue

                try:
                    c_fold = float(np.corrcoef(x_tr_clean, y_tr_clean)[0, 1])
                except Exception:
                    continue

                if np.isfinite(c_fold):
                    fold_scores.append(abs(c_fold))

            if fold_scores:
                m = float(np.mean(fold_scores))
                s = float(np.std(fold_scores))
                mi_mean[col] = m
                mi_cv[col] = float(s / max(abs(m), 1e-12))
            else:
                mi_mean[col] = 0.0
                mi_cv[col] = float("inf")
    else:
        # Continuous/real-valued target: use event-aware MI proxy and stability utilities
        mi_full = component._event_aware_feature_scores(X, y).fillna(0.0)

        mi_stab = component.calculate_mi_stability(
            X=X,
            y=y,
            selected_features=list(X.columns),
            cv_folds=cv_folds,
        )

        mi_mean: Dict[str, float] = mi_stab.get("mi_mean", {}) if isinstance(mi_stab, dict) else {}
        mi_cv: Dict[str, float] = mi_stab.get("mi_cv", {}) if isinstance(mi_stab, dict) else {}

    # Simple Pearson correlation and R^2 per feature
    # (uses the same cleaned y_arr computed above)
    metrics: Dict[str, Dict[str, float]] = {}

    for col in X.columns:
        x = X[col].to_numpy(dtype=float)

        # Skip only fully-missing features; allow constant ones so we still
        # see them in the report (corr/R² will just be NaN).
        if np.all(np.isnan(x)):
            continue

        try:
            corr = float(np.corrcoef(x, y_arr)[0, 1])
        except Exception:
            corr = float("nan")

        if not np.isfinite(corr):
            r2 = float("nan")
        else:
            r2 = float(corr ** 2)

        metrics[col] = {
            "mi_proxy_full": float(mi_full.get(col, 0.0)),
            "mi_mean_cv": float(mi_mean.get(col, 0.0)),
            "mi_cv": float(mi_cv.get(col, float("inf"))),
            "pearson_corr": corr,
            "r2": r2,
        }

    return metrics


def _infer_model_group(feature_name: str) -> str:
    name = feature_name.lower()
    if name.startswith("path_risk") or "path_risk" in name:
        return "path_risk"
    if name.startswith("risk_regime") or name.startswith("risk_pred") or name.startswith("risk_") or "risk_regime" in name:
        return "risk"
    if name.startswith("smc_"):
        return "smc"
    if name.startswith("macro_trend") or (name.startswith("macro_") and "alpha" in name):
        return "macro_trend"
    if "meso_trend" in name:
        return "meso_trend"
    if name.startswith("liquidity_regime") or "liquidity" in name:
        return "liquidity"
    if name.startswith("breakout_") or name in {"is_resistance", "is_support", "resistance_scalar", "support_scalar"}:
        return "breakout_bounce"
    if name.startswith("mr_") or "mean_reversion" in name:
        return "mean_reversion"
    return "other"


def _select_specialist_scalars(X: pd.DataFrame) -> pd.DataFrame:
    """Collapse raw specialist outputs to canonical scalar features for diagnostics.

    This function is diagnostics-only: it does not modify artifacts, only the
    feature matrix used by specialist_feature_diagnostics. The goals are:

    - Risk:    single scalar risk_score in [0, 1]
    - Liquidity: keep regime probabilities as-is
    - Breakout: two support scalars, two resistance scalars
    - Path/macro: single scalar per path/macro specialist if available
    - SMC:     single scalar smc_predicted
    - MR:      single scalar mean-reversion score from XGB (mr_probability/raw)
    """

    X = X.copy()
    cols = list(X.columns)

    # ------------------------------------------------------------------
    # Risk: prefer explicit risk_score, otherwise derive from risk_regime
    # ------------------------------------------------------------------
    if "risk_score" not in cols and "risk_regime" in cols:
        rr = X["risk_regime"].astype(float)
        max_rr = float(np.nanmax(rr)) if rr.notna().any() else 0.0
        if np.isfinite(max_rr) and max_rr > 0.0:
            X["risk_score"] = rr / max_rr
        else:
            X["risk_score"] = 0.0
        cols = list(X.columns)

    risk_cols = [c for c in cols if c.startswith("risk_")]
    if "risk_score" in X.columns:
        risk_keep = {"risk_score"}
        risk_drop = [c for c in risk_cols if c not in risk_keep]
        X = X.drop(columns=risk_drop, errors="ignore")
        cols = list(X.columns)

    # ------------------------------------------------------------------
    # Breakout/Bounce: keep 2 support + 2 resistance scalars
    # ------------------------------------------------------------------
    breakout_keep = set()
    for c in (
        "resistance_scalar",
        "breakout_scalar_resistance",
        "support_scalar",
        "breakout_scalar_support",
        "breakout_success_prob",
        "breakout_high_conf_signal",
    ):
        if c in X.columns:
            breakout_keep.add(c)

    breakout_cols = [
        c
        for c in cols
        if c.startswith("breakout_") or c in {"is_resistance", "is_support"}
    ]
    breakout_drop = [c for c in breakout_cols if c not in breakout_keep]
    if breakout_drop:
        X = X.drop(columns=breakout_drop, errors="ignore")
        cols = list(X.columns)

    # ------------------------------------------------------------------
    # Path: prefer dedicated risk-style scalar if present
    # ------------------------------------------------------------------
    path_cols = [c for c in cols if c.startswith("path_")]
    path_scalar_col: Optional[str] = None
    if "path_risk_score" in X.columns:
        path_scalar_col = "path_risk_score"
    elif "path_regime" in X.columns:
        pr = X["path_regime"].astype(float)
        max_pr = float(np.nanmax(pr)) if pr.notna().any() else 0.0
        if np.isfinite(max_pr) and max_pr > 0.0:
            X["path_risk_score"] = pr / max_pr
        else:
            X["path_risk_score"] = 0.0
        path_scalar_col = "path_risk_score"

    if path_scalar_col is not None:
        path_keep = {path_scalar_col}
        path_drop = [c for c in path_cols if c not in path_keep]
        X = X.drop(columns=path_drop, errors="ignore")
        cols = list(X.columns)

    # SMC: keep a single scalar (smc_predicted) and drop auxiliary columns
    smc_cols = [c for c in cols if c.startswith("smc_")]
    smc_keep = set()
    if "smc_predicted" in X.columns:
        smc_keep.add("smc_predicted")

    smc_drop = [c for c in smc_cols if c not in smc_keep]
    if smc_drop:
        X = X.drop(columns=smc_drop, errors="ignore")
        cols = list(X.columns)

    # ------------------------------------------------------------------
    # Mean reversion: keep a single XGB score, preferring dense
    # probabilities when available.
    # ------------------------------------------------------------------
    mr_cols = [c for c in cols if c.startswith("mr_")]
    mr_keep: set[str] = set()
    if "mr_probability_dense" in X.columns:
        mr_keep.add("mr_probability_dense")
    elif "mr_probability" in X.columns:
        mr_keep.add("mr_probability")
    elif "mr_raw_score" in X.columns:
        mr_keep.add("mr_raw_score")

    mr_drop = [c for c in mr_cols if c not in mr_keep]
    if mr_drop:
        X = X.drop(columns=mr_drop, errors="ignore")

    return X


def _compute_model_reliability(
    feature_metrics: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    groups: Dict[str, Dict[str, Any]] = {}
    for feat, met in feature_metrics.items():
        group = _infer_model_group(feat)
        if group == "other":
            continue
        g = groups.setdefault(
            group,
            {
                "features": [],
                "mi_values": [],
                "r2_values": [],
            },
        )
        mi_val = float(met.get("mi_mean_cv", 0.0))
        r2_val = float(met.get("r2", 0.0))
        g["features"].append(feat)
        if np.isfinite(mi_val):
            g["mi_values"].append(mi_val)
        if np.isfinite(r2_val):
            g["r2_values"].append(r2_val)

    per_model: Dict[str, Dict[str, float]] = {}
    for group, data in groups.items():
        mi_arr = np.array(data["mi_values"], dtype=float)
        r2_arr = np.array(data["r2_values"], dtype=float)
        n_features = len(data["features"])
        model_summary: Dict[str, float] = {
            "n_features": int(n_features),
            "mi_mean_avg": float(mi_arr.mean()) if mi_arr.size else 0.0,
            "mi_mean_median": float(np.median(mi_arr)) if mi_arr.size else 0.0,
            "r2_mean": float(r2_arr.mean()) if r2_arr.size else 0.0,
            "r2_median": float(np.median(r2_arr)) if r2_arr.size else 0.0,
            "n_high_mi": int(np.sum(mi_arr > 0.1)) if mi_arr.size else 0,
            "n_high_r2": int(np.sum(r2_arr > 0.05)) if r2_arr.size else 0,
        }

        best_mi: Optional[float] = None
        best_mi_feat: Optional[str] = None
        best_r2: Optional[float] = None
        best_r2_feat: Optional[str] = None

        for feat in data["features"]:
            met = feature_metrics.get(feat, {})
            mi_val = float(met.get("mi_mean_cv", 0.0))
            r2_val = float(met.get("r2", 0.0))
            if np.isfinite(mi_val) and (best_mi is None or mi_val > best_mi):
                best_mi = mi_val
                best_mi_feat = feat
            if np.isfinite(r2_val) and (best_r2 is None or r2_val > best_r2):
                best_r2 = r2_val
                best_r2_feat = feat

        if best_mi is not None:
            model_summary["best_mi_feature_value"] = float(best_mi)
        if best_mi_feat is not None:
            model_summary["best_mi_feature"] = best_mi_feat
        if best_r2 is not None:
            model_summary["best_r2_feature_value"] = float(best_r2)
        if best_r2_feat is not None:
            model_summary["best_r2_feature"] = best_r2_feat

        per_model[group] = model_summary

    ranked_by_mi = sorted(
        per_model.items(), key=lambda kv: kv[1].get("mi_mean_avg", 0.0), reverse=True
    )
    ranked_by_r2 = sorted(
        per_model.items(), key=lambda kv: kv[1].get("r2_mean", 0.0), reverse=True
    )

    return {
        "per_model": per_model,
        "ranked_by_mi": [g for g, _ in ranked_by_mi],
        "ranked_by_r2": [g for g, _ in ranked_by_r2],
    }


def _compute_model_coverage(
    X: pd.DataFrame,
    y: pd.Series,
) -> Dict[str, Any]:
    """Compute data coverage (date range and sample count) per specialist group.

    Coverage is defined over the intersection of the specialist feature index
    and the target index. For each specialist group (alpha, macro_trend,
    liquidity, breakout_bounce, risk, smc, mean_reversion), we report:

    - n_samples: number of target samples where at least one feature in that
      group is non-null
    - start / end: first and last timestamps (on the target index) where the
      group has any non-null coverage
    """
    coverage: Dict[str, Dict[str, Any]] = {}

    if not isinstance(X.index, pd.DatetimeIndex):
        return coverage

    common_index = X.index.intersection(y.index)
    if len(common_index) == 0:
        return coverage

    Xc = X.loc[common_index]
    yc = y.loc[common_index]

    # Initialize groups based on features actually present
    for feat in Xc.columns:
        group = _infer_model_group(feat)
        if group == "other":
            continue
        coverage.setdefault(
            group,
            {
                "n_samples": 0,
                "start": None,
                "end": None,
                "fraction_of_target": 0.0,
            },
        )

    if not coverage:
        return coverage

    total = int(len(yc)) if len(yc) > 0 else 0

    for group, info in coverage.items():
        group_cols = [c for c in Xc.columns if _infer_model_group(c) == group]
        if not group_cols:
            continue
        mask = (~Xc[group_cols].isna()).any(axis=1)
        if not mask.any():
            continue
        idx = yc.index[mask]
        if len(idx) == 0:
            continue

        n_samples = int(mask.sum())
        info["n_samples"] = n_samples
        info["start"] = idx.min()
        info["end"] = idx.max()
        info["fraction_of_target"] = float(n_samples / total) if total > 0 else 0.0

    return coverage


def _compute_model_pairwise_relationships(
    X: pd.DataFrame,
    feature_metrics: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    group_to_best: Dict[str, str] = {}
    group_to_best_score: Dict[str, float] = {}

    for feat, met in feature_metrics.items():
        group = _infer_model_group(feat)
        if group == "other":
            continue
        score = float(met.get("mi_mean_cv", 0.0))
        prev = group_to_best_score.get(group)
        if prev is None or score > prev:
            group_to_best_score[group] = score
            group_to_best[group] = feat

    if len(group_to_best) < 2:
        return {"error": "Not enough specialist model groups for pairwise analysis"}

    reps: Dict[str, pd.Series] = {}
    for group, feat in group_to_best.items():
        if feat not in X.columns:
            continue
        s = X[feat].astype(float).replace([np.inf, -np.inf], np.nan)
        reps[group] = s

    if len(reps) < 2:
        return {"error": "Representative features missing in X for pairwise analysis"}

    common_index: Optional[pd.DatetimeIndex] = None
    for s in reps.values():
        if common_index is None:
            common_index = s.index
        else:
            common_index = common_index.intersection(s.index)

    if common_index is None or len(common_index) == 0:
        return {"error": "No overlapping samples for pairwise analysis"}

    matrix = pd.DataFrame(
        {g: s.loc[common_index] for g, s in reps.items()},
        index=common_index,
    )
    matrix = matrix.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    config = FinalFeatureSelectionConfig()
    component = FinalFeatureSelectionComponent(config=config)

    groups = sorted(matrix.columns)
    pairwise: list[Dict[str, Any]] = []

    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            gi = groups[i]
            gj = groups[j]

            xi = matrix[gi].to_numpy(dtype=float)
            xj = matrix[gj].to_numpy(dtype=float)

            try:
                corr = float(np.corrcoef(xi, xj)[0, 1])
            except Exception:
                corr = float("nan")

            if np.isfinite(corr):
                r2_val = float(corr ** 2)
            else:
                r2_val = float("nan")

            df_i = pd.DataFrame({gi: matrix[gi]})
            df_j = pd.DataFrame({gj: matrix[gj]})

            try:
                mi_forward_scores = component._event_aware_feature_scores(df_i, matrix[gj])
                mi_forward = float(mi_forward_scores.get(gi, 0.0))
            except Exception:
                mi_forward = 0.0

            try:
                mi_backward_scores = component._event_aware_feature_scores(df_j, matrix[gi])
                mi_backward = float(mi_backward_scores.get(gj, 0.0))
            except Exception:
                mi_backward = 0.0

            mi_sym = 0.5 * (mi_forward + mi_backward)

            pairwise.append(
                {
                    "model_i": gi,
                    "model_j": gj,
                    "rep_feature_i": group_to_best.get(gi),
                    "rep_feature_j": group_to_best.get(gj),
                    "mi_proxy": float(mi_sym),
                    "mi_forward": float(mi_forward),
                    "mi_backward": float(mi_backward),
                    "r2": r2_val,
                }
            )

    pairwise.sort(key=lambda d: d["mi_proxy"], reverse=True)

    return {
        "representatives": group_to_best,
        "pairs": pairwise,
    }


def _compute_probe_models(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> Dict[str, Any]:
    """Fit simple probe models (LogReg, LightGBM) and report metrics.

    For now this assumes a binary meta-label target (0/1). If the target is
    not approximately binary, the function returns a descriptive payload and
    skips model fitting.
    """
    # Align and clean
    common_index = X.index.intersection(y.index)
    Xc = X.loc[common_index].copy()
    yc = y.loc[common_index].astype(float)

    mask = ~yc.isna()
    Xc = Xc.loc[mask]
    yc = yc.loc[mask]

    # Ensure there are no NaNs in X for sklearn probe models
    Xc = Xc.fillna(0.0)

    result: Dict[str, Any] = {
        "n_samples": int(len(yc)),
        "n_features": int(Xc.shape[1]),
        "task_type": "unknown",
    }

    if len(yc) < max(100, n_splits * 10):
        result["warning"] = (
            "Very few samples for probe models; metrics may be unstable"
        )

    # Determine if target looks binary
    uniq = np.unique(yc.values[~np.isnan(yc.values)])
    if uniq.size == 0:
        result["error"] = "No valid target samples for probe models"
        return result

    # Round to nearest integer and check if values are in {0,1}
    uniq_rounded = np.unique(np.round(uniq).astype(int))
    if not set(uniq_rounded).issubset({0, 1}):
        result["task_type"] = "non_binary_target"
        result["error"] = (
            "Probe models currently implemented only for binary targets"
        )
        return result

    # Binary classification setup
    y_bin = (yc > 0.5).astype(int)
    pos_frac = float(y_bin.mean())
    result["task_type"] = "binary_classification"
    result["class_balance"] = {"pos_frac": pos_frac, "neg_frac": 1.0 - pos_frac}

    tscv = TimeSeriesSplit(n_splits=n_splits)

    def _collect_scores(probs: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        # Metrics based on predicted probabilities
        auc = roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else np.nan
        acc = accuracy_score(y_true, (probs >= 0.5).astype(int))
        brier = brier_score_loss(y_true, probs)
        # Older sklearn versions don't support 'squared' kwarg; compute RMSE
        mse = mean_squared_error(y_true, probs)
        rmse = float(np.sqrt(mse)) if np.isfinite(mse) else float("nan")
        r2 = r2_score(y_true, probs)
        return {
            "auc": float(auc) if np.isfinite(auc) else float("nan"),
            "accuracy": float(acc),
            "brier": float(brier),
            "rmse": float(rmse),
            "pseudo_r2": float(r2),
        }

    # Logistic Regression probe
    logreg_scores: Dict[str, list[float]] = {
        "auc": [],
        "accuracy": [],
        "brier": [],
        "rmse": [],
        "pseudo_r2": [],
    }

    for train_idx, test_idx in tscv.split(Xc):
        X_tr, X_te = Xc.iloc[train_idx], Xc.iloc[test_idx]
        y_tr, y_te = y_bin.iloc[train_idx], y_bin.iloc[test_idx]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=200,
                        n_jobs=-1,
                        class_weight="balanced",
                    ),
                ),
            ]
        )
        pipe.fit(X_tr, y_tr)
        p_te = pipe.predict_proba(X_te)[:, 1]
        fold = _collect_scores(p_te, y_te.values)
        for k, v in fold.items():
            if np.isfinite(v):
                logreg_scores[k].append(v)

    if any(logreg_scores.values()):
        result["logreg"] = {
            f"{k}_mean": float(np.mean(v)) if v else float("nan"),
            f"{k}_std": float(np.std(v)) if v else float("nan"),
        }
        for k, v in logreg_scores.items():
            result["logreg"][f"{k}_mean"] = float(np.mean(v)) if v else float("nan")
            result["logreg"][f"{k}_std"] = float(np.std(v)) if v else float("nan")

    # LightGBM probe (optional)
    try:
        import lightgbm as lgb  # type: ignore

        lgbm_scores: Dict[str, list[float]] = {
            "auc": [],
            "accuracy": [],
            "brier": [],
            "rmse": [],
            "pseudo_r2": [],
        }
        for train_idx, test_idx in tscv.split(Xc):
            X_tr, X_te = Xc.iloc[train_idx], Xc.iloc[test_idx]
            y_tr, y_te = y_bin.iloc[train_idx], y_bin.iloc[test_idx]
            if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                continue
            model = lgb.LGBMClassifier(
                objective="binary",
                n_estimators=200,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X_tr, y_tr)
            p_te = model.predict_proba(X_te)[:, 1]
            fold = _collect_scores(p_te, y_te.values)
            for k, v in fold.items():
                if np.isfinite(v):
                    lgbm_scores[k].append(v)

        if any(lgbm_scores.values()):
            result["lgbm"] = {}
            for k, v in lgbm_scores.items():
                result["lgbm"][f"{k}_mean"] = float(np.mean(v)) if v else float("nan")
                result["lgbm"][f"{k}_std"] = float(np.std(v)) if v else float("nan")
    except ImportError:
        result["lgbm"] = {"error": "lightgbm not available"}

    return result


def _detect_feature_leakage(
    X: pd.DataFrame,
    y: pd.Series,
    component: FinalFeatureSelectionComponent,
) -> Dict[str, Any]:
    """Use FinalFeatureSelectionComponent's leakage detector on specialist features."""
    try:
        leakage = component.detect_potential_leakage(
            X=X,
            y=y,
            selected_features=list(X.columns),
        )
        return leakage
    except Exception as exc:
        logger.warning("Leakage detection failed: %s", exc)
        return {"error": str(exc)}


def _compute_global_stability(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> Dict[str, Any]:
    """Compute a simple global stability metric via CV AUC variability."""
    common_index = X.index.intersection(y.index)
    Xc = X.loc[common_index].copy()
    yc = y.loc[common_index].astype(float)
    mask = ~yc.isna()
    Xc = Xc.loc[mask]
    yc = yc.loc[mask]

    # Ensure there are no NaNs in X for sklearn probe model
    Xc = Xc.fillna(0.0)

    # Assume binary labels for now
    y_bin = (yc > 0.5).astype(int)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    aucs: list[float] = []
    for train_idx, test_idx in tscv.split(Xc):
        X_tr, X_te = Xc.iloc[train_idx], Xc.iloc[test_idx]
        y_tr, y_te = y_bin.iloc[train_idx], y_bin.iloc[test_idx]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=200, n_jobs=-1, class_weight="balanced")),
            ]
        )
        pipe.fit(X_tr, y_tr)
        p_te = pipe.predict_proba(X_te)[:, 1]
        if len(np.unique(y_te)) < 2:
            continue
        try:
            auc = roc_auc_score(y_te.values, p_te)
            if np.isfinite(auc):
                aucs.append(float(auc))
        except Exception:
            continue

    if not aucs:
        return {"error": "Insufficient folds for stability analysis"}

    mean_auc = float(np.mean(aucs))
    std_auc = float(np.std(aucs))
    stability = float(1.0 - std_auc / mean_auc) if mean_auc > 0 else float("nan")
    return {
        "n_splits": int(n_splits),
        "fold_aucs": aucs,
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "stability_score": stability,
    }


def _compute_tree_shap_interactions(
    X: pd.DataFrame,
    y: pd.Series,
    feature_metrics: Dict[str, Dict[str, float]],
    max_features: int = 30,
    sample_size: int = 2000,
) -> Dict[str, Any]:
    """Estimate notable pairwise interactions using TreeSHAP (if available)."""
    try:
        import lightgbm as lgb  # type: ignore
        import shap  # type: ignore
    except ImportError as exc:
        return {"error": f"lightgbm or shap not available: {exc}"}

    # Align and clean
    common_index = X.index.intersection(y.index)
    Xc = X.loc[common_index].copy()
    yc = y.loc[common_index].astype(float)
    mask = ~yc.isna()
    Xc = Xc.loc[mask]
    yc = yc.loc[mask]

    # Binary labels assumed
    y_bin = (yc > 0.5).astype(int)

    # Select top features by MI_mean (or fall back to all)
    if feature_metrics:
        ranked = sorted(
            feature_metrics.items(),
            key=lambda kv: kv[1].get("mi_mean_cv", 0.0),
            reverse=True,
        )
        top_names = [name for name, _ in ranked[:max_features]]
    else:
        top_names = list(Xc.columns)[:max_features]

    X_sel = Xc[top_names].copy()

    # Subsample for SHAP efficiency (use latest samples chronologically)
    if len(X_sel) > sample_size:
        X_sample = X_sel.iloc[-sample_size:]
        y_sample = y_bin.iloc[-sample_size:]
    else:
        X_sample = X_sel
        y_sample = y_bin

    if X_sample.empty or len(np.unique(y_sample)) < 2:
        return {"error": "Insufficient data for TreeSHAP interactions"}

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_sample, y_sample)

    try:
        explainer = shap.TreeExplainer(model)
        shap_int = explainer.shap_interaction_values(X_sample)
    except Exception as exc:
        return {"error": f"TreeSHAP interaction computation failed: {exc}"}

    # shap_int shape: (n_samples, n_features, n_features)
    if isinstance(shap_int, list):
        # For binary classification, shap returns list per class; use average
        shap_int_arr = np.mean(np.abs(np.array(shap_int)), axis=0)
    else:
        shap_int_arr = np.mean(np.abs(shap_int), axis=0)

    n_feat = shap_int_arr.shape[0]
    pairs: list[Dict[str, Any]] = []
    for i in range(n_feat):
        for j in range(i + 1, n_feat):
            score = float(shap_int_arr[i, j])
            pairs.append(
                {
                    "feature_i": top_names[i],
                    "feature_j": top_names[j],
                    "interaction_strength": score,
                }
            )

    if not pairs:
        return {"error": "No interaction pairs computed"}

    pairs.sort(key=lambda d: d["interaction_strength"], reverse=True)
    top_pairs = pairs[:20]

    return {
        "n_features": len(top_names),
        "sample_size": int(len(X_sample)),
        "top_pairs": top_pairs,
        "method": "tree_shap",
    }


def _compute_trading_simulation_pnl(
    confidence_scores: pd.Series,
    realized_returns: pd.Series,
    thresholds: list[float] = [0.6, 0.7, 0.8, 0.9],
    tp_pct: float = 0.02,  # 2% take profit
    sl_pct: float = 0.007,  # 0.7% stop loss
    fee_per_trade: float = 0.0015,  # 0.15% per trade (0.3% round trip)
) -> Dict[str, Any]:
    """Compute trading simulation PnL for specialist/probe models at various confidence thresholds.
    
    Uses a simplified triple-barrier approach:
    - TP: Take profit at +2%
    - SL: Stop loss at -0.7%
    - Fees: 0.15% per trade (0.3% round-trip)
    
    For each threshold, computes:
    - Number of trades
    - Average trades per day
    - Win rate
    - Average PnL % per-day and per-month
    
    Args:
        confidence_scores: Model confidence/probability scores (0-1)
        realized_returns: Actual realized returns for each event
        thresholds: Confidence thresholds to evaluate
        tp_pct: Take profit percentage (default: 2%)
        sl_pct: Stop loss percentage (default: 0.7%)
        fee_per_trade: Fee per trade as fraction (default: 0.15%)
        
    Returns:
        Dictionary with results per threshold
    """
    results = {
        "thresholds": thresholds,
        "tp_pct": tp_pct,
        "sl_pct": sl_pct,
        "fee_round_trip_pct": fee_per_trade * 2,
        "per_threshold": {},
    }
    
    # Align confidence and returns
    common_idx = confidence_scores.index.intersection(realized_returns.index)
    if len(common_idx) < 10:
        return {"error": f"Insufficient overlapping data: {len(common_idx)} samples"}
    
    conf = confidence_scores.loc[common_idx].astype(float)
    rets = realized_returns.loc[common_idx].astype(float)
    
    # Drop NaN values
    valid_mask = conf.notna() & rets.notna()
    conf = conf[valid_mask]
    rets = rets[valid_mask]
    
    if len(conf) < 10:
        return {"error": f"Insufficient valid data after NaN removal: {len(conf)} samples"}
    
    # Calculate date range for per-day/per-month metrics
    if isinstance(conf.index, pd.DatetimeIndex):
        date_range_days = (conf.index.max() - conf.index.min()).days
        if date_range_days <= 0:
            date_range_days = 1
    else:
        date_range_days = len(conf) / 96  # Assume 15m bars, ~96 per day
    
    date_range_months = date_range_days / 30.44  # Average days per month
    
    for threshold in thresholds:
        # Filter trades above threshold
        trade_mask = conf >= threshold
        n_trades = int(trade_mask.sum())
        
        if n_trades == 0:
            results["per_threshold"][f"{threshold:.0%}"] = {
                "threshold": threshold,
                "n_trades": 0,
                "trades_per_day": 0.0,
                "win_rate": 0.0,
                "total_pnl_pct": 0.0,
                "avg_pnl_per_trade_pct": 0.0,
                "avg_pnl_per_day_pct": 0.0,
                "avg_pnl_per_month_pct": 0.0,
                "sharpe_estimate": 0.0,
            }
            continue
        
        # Get returns for trades above threshold
        trade_returns = rets[trade_mask]
        
        # Apply triple-barrier logic (simplified):
        # If return >= tp_pct: profit = tp_pct - fees
        # If return <= -sl_pct: loss = -sl_pct - fees  
        # Otherwise: use actual return - fees
        def apply_barrier(ret: float) -> float:
            if ret >= tp_pct:
                return tp_pct - fee_per_trade * 2  # Hit TP
            elif ret <= -sl_pct:
                return -sl_pct - fee_per_trade * 2  # Hit SL
            else:
                return ret - fee_per_trade * 2  # Actual return minus fees
        
        barrier_returns = trade_returns.apply(apply_barrier)
        
        # Calculate metrics
        n_wins = int((barrier_returns > 0).sum())
        win_rate = n_wins / n_trades if n_trades > 0 else 0.0
        
        total_pnl_pct = float(barrier_returns.sum()) * 100  # As percentage
        avg_pnl_per_trade_pct = float(barrier_returns.mean()) * 100
        
        trades_per_day = n_trades / date_range_days if date_range_days > 0 else 0.0
        avg_pnl_per_day_pct = total_pnl_pct / date_range_days if date_range_days > 0 else 0.0
        avg_pnl_per_month_pct = total_pnl_pct / date_range_months if date_range_months > 0 else 0.0
        
        # Estimate Sharpe-like metric (annualized)
        if len(barrier_returns) > 1 and barrier_returns.std() > 0:
            daily_sharpe = (barrier_returns.mean() / barrier_returns.std()) * np.sqrt(trades_per_day)
            annualized_sharpe = daily_sharpe * np.sqrt(252)
        else:
            annualized_sharpe = 0.0
        
        results["per_threshold"][f"{threshold:.0%}"] = {
            "threshold": threshold,
            "n_trades": n_trades,
            "trades_per_day": round(trades_per_day, 2),
            "win_rate": round(win_rate, 4),
            "total_pnl_pct": round(total_pnl_pct, 2),
            "avg_pnl_per_trade_pct": round(avg_pnl_per_trade_pct, 4),
            "avg_pnl_per_day_pct": round(avg_pnl_per_day_pct, 4),
            "avg_pnl_per_month_pct": round(avg_pnl_per_month_pct, 2),
            "sharpe_estimate": round(annualized_sharpe, 2),
        }
    
    results["date_range_days"] = round(date_range_days, 1)
    results["total_samples"] = len(conf)
    
    return results


def _compute_probe_model_pnl(
    X: pd.DataFrame,
    y: pd.Series,
    realized_returns: pd.Series,
    n_splits: int = 5,
    thresholds: list[float] = [0.6, 0.7, 0.8, 0.9],
) -> Dict[str, Any]:
    """Compute trading PnL metrics for probe models (LogReg/LGBM) using cross-validation predictions.
    
    This function trains probe models with TimeSeriesSplit cross-validation,
    collects out-of-sample probability predictions, and computes trading PnL
    at various confidence thresholds.
    """
    results = {}
    
    # Check for valid data
    common_idx = X.index.intersection(y.index).intersection(realized_returns.index)
    if len(common_idx) < 100:
        return {"error": f"Insufficient overlapping data: {len(common_idx)} samples"}
    
    X_aligned = X.loc[common_idx].copy()
    y_aligned = y.loc[common_idx].copy()
    rets_aligned = realized_returns.loc[common_idx].copy()
    
    # Drop NaN values
    valid_mask = y_aligned.notna() & rets_aligned.notna()
    X_aligned = X_aligned.loc[valid_mask]
    y_aligned = y_aligned[valid_mask]
    rets_aligned = rets_aligned[valid_mask]
    
    # Fill NaN in features
    X_aligned = X_aligned.fillna(0.0)
    
    if len(X_aligned) < 100:
        return {"error": f"Insufficient valid data: {len(X_aligned)} samples"}
    
    # Convert target to binary (for classification)
    y_binary = (y_aligned > 0).astype(int)
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Collect out-of-sample predictions
    oos_probs_logreg = pd.Series(index=X_aligned.index, dtype=float)
    oos_probs_lgbm = pd.Series(index=X_aligned.index, dtype=float)
    
    for train_idx, val_idx in tscv.split(X_aligned):
        X_train = X_aligned.iloc[train_idx]
        X_val = X_aligned.iloc[val_idx]
        y_train = y_binary.iloc[train_idx]
        
        # Logistic Regression
        try:
            logreg = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=500, solver="lbfgs", random_state=42)),
            ])
            logreg.fit(X_train, y_train)
            probs = logreg.predict_proba(X_val)[:, 1]
            oos_probs_logreg.iloc[val_idx] = probs
        except Exception:
            pass
        
        # LightGBM
        try:
            import lightgbm as lgb
            lgbm_clf = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.05,
                random_state=42,
                verbose=-1,
            )
            lgbm_clf.fit(X_train, y_train)
            probs = lgbm_clf.predict_proba(X_val)[:, 1]
            oos_probs_lgbm.iloc[val_idx] = probs
        except Exception:
            pass
    
    # Compute trading PnL for each model
    if oos_probs_logreg.notna().sum() >= 10:
        results["logreg"] = _compute_trading_simulation_pnl(
            confidence_scores=oos_probs_logreg,
            realized_returns=rets_aligned,
            thresholds=thresholds,
        )
    else:
        results["logreg"] = {"error": "Insufficient LogReg predictions"}
    
    if oos_probs_lgbm.notna().sum() >= 10:
        results["lgbm"] = _compute_trading_simulation_pnl(
            confidence_scores=oos_probs_lgbm,
            realized_returns=rets_aligned,
            thresholds=thresholds,
        )
    else:
        results["lgbm"] = {"error": "Insufficient LGBM predictions"}
    
    return results


def run_diagnostics(
    symbol: str,
    exchange: str,
    timeframe: str,
    direction: str,
    model: str,
    regime_timeframe: str,
    target_col: str,
    cv_folds: int,
    lookback_days: Optional[float] = None,
    enable_risk_hmm_specialist: bool = True,
    projection_mode: str = "canonical_scalars",
) -> Tuple[Path, Path]:
    """Run full specialist feature diagnostics and export reports."""
    # 1) Load labels and realized_return for PnL simulation
    y, training_index, realized_return = _prepare_labels(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        direction=direction,
        model=model,
        target_col=target_col,
        lookback_days=lookback_days,
    )

    # 2) Load specialist features aligned to the same index
    X_raw = _load_specialist_features(
        symbol=symbol,
        exchange=exchange,
        base_timeframe=timeframe,
        regime_timeframe=regime_timeframe,
        direction=direction,
        model=model,
        training_index=training_index,
        enable_risk_hmm_specialist=enable_risk_hmm_specialist,
    )

    # Optionally collapse raw specialist outputs to canonical scalars. When
    # projection_mode == "raw", use the specialist block exactly as training
    # sees it (same get_specialist_models_outputs output).
    if projection_mode == "canonical_scalars":
        X = _select_specialist_scalars(X_raw)
    else:
        X = X_raw

    # 3) Compute per-feature metrics
    feature_metrics = _compute_feature_metrics(X=X, y=y, cv_folds=cv_folds)

    if not feature_metrics:
        raise ValueError("No feature metrics computed; check inputs and artifacts")

    model_reliability = _compute_model_reliability(feature_metrics=feature_metrics)
    model_coverage = _compute_model_coverage(X=X, y=y)
    model_relationships = _compute_model_pairwise_relationships(X=X, feature_metrics=feature_metrics)

    # 3b) Compute target date range
    y_start = y.index.min()
    y_end = y.index.max()
    y_duration = (y_end - y_start).days if len(y) > 0 else 0

    # 4) Probe models (LogReg / LGBM), leakage, stability, interactions
    probe_models = _compute_probe_models(X=X, y=y, n_splits=cv_folds)
    
    # 4b) Compute probe model trading PnL simulation
    probe_pnl = _compute_probe_model_pnl(
        X=X,
        y=y,
        realized_returns=realized_return,
        n_splits=cv_folds,
        thresholds=[0.6, 0.7, 0.8, 0.9],
    )

    # Reuse a FinalFeatureSelectionComponent instance for leakage detection
    fs_config = FinalFeatureSelectionConfig()
    fs_component = FinalFeatureSelectionComponent(config=fs_config)
    leakage_diagnostics = _detect_feature_leakage(X=X, y=y, component=fs_component)

    global_stability = _compute_global_stability(X=X, y=y, n_splits=cv_folds)
    interactions = _compute_tree_shap_interactions(
        X=X,
        y=y,
        feature_metrics=feature_metrics,
    )

    # Aggregate summary stats
    mi_values = np.array([m["mi_mean_cv"] for m in feature_metrics.values()], dtype=float)
    r2_values = np.array([m["r2"] for m in feature_metrics.values()], dtype=float)

    mi_values = mi_values[np.isfinite(mi_values)]
    r2_values = r2_values[np.isfinite(r2_values)]

    summary: Dict[str, Any] = {
        "n_features": len(feature_metrics),
        "mi_mean_avg": float(mi_values.mean()) if mi_values.size else 0.0,
        "mi_mean_median": float(np.median(mi_values)) if mi_values.size else 0.0,
        "r2_mean": float(r2_values.mean()) if r2_values.size else 0.0,
        "r2_median": float(np.median(r2_values)) if r2_values.size else 0.0,
        "n_high_mi": int(np.sum(mi_values > 0.1)) if mi_values.size else 0,
        "n_high_r2": int(np.sum(r2_values > 0.05)) if r2_values.size else 0,
    }

    # Build Markdown summary (top features by MI and R^2)
    sorted_by_mi = sorted(
        feature_metrics.items(), key=lambda kv: kv[1]["mi_mean_cv"], reverse=True
    )
    sorted_by_r2 = sorted(
        feature_metrics.items(), key=lambda kv: kv[1]["r2"], reverse=True
    )

    top_k = 20
    md_lines: list[str] = [
        "# Specialist Feature Diagnostics",
        "",
        f"**Symbol**: {symbol}",
        f"**Exchange**: {exchange}",
        f"**Timeframe**: {timeframe}",
        f"**Direction**: {direction}",
        f"**Model**: {model}",
        f"**Regime timeframe**: {regime_timeframe}",
        f"**Target column**: {target_col}",
        "",
        "## Data Range Analysis",
        f"- Target start date: {y_start}",
        f"- Target end date: {y_end}",
        f"- Target duration: {y_duration} days",
        f"- Target samples: {len(y)}",
        "",
        "## Overview",
        f"- Number of specialist features: {summary['n_features']}",
        f"- Mean MI (CV-averaged): {summary['mi_mean_avg']:.4f}",
        f"- Median MI (CV-averaged): {summary['mi_mean_median']:.4f}",
        f"- Mean R^2 (univariate): {summary['r2_mean']:.4f}",
        f"- Median R^2 (univariate): {summary['r2_median']:.4f}",
        f"- High-MI features (MI>0.10): {summary['n_high_mi']}",
        f"- High-R^2 features (R^2>0.05): {summary['n_high_r2']}",
        "",
        "### Probe model summary (LogReg / LGBM)",
    ]

    # Add brief probe model summary if available
    logreg_summary = probe_models.get("logreg", {}) if isinstance(probe_models, dict) else {}
    lgbm_summary = probe_models.get("lgbm", {}) if isinstance(probe_models, dict) else {}

    def _fmt_probe(model_name: str, summary_dict: Dict[str, Any]) -> str:
        if not summary_dict or "auc_mean" not in summary_dict:
            return f"- {model_name}: not available"
        auc_mean = summary_dict.get("auc_mean", float("nan"))
        auc_std = summary_dict.get("auc_std", float("nan"))
        acc_mean = summary_dict.get("accuracy_mean", float("nan"))
        return (
            f"- {model_name}: AUC={auc_mean:.3f}±{auc_std:.3f}, "
            f"Accuracy={acc_mean:.3f}"
        )

    md_lines.extend(
        [
            _fmt_probe("Logistic Regression", logreg_summary),
            _fmt_probe("LightGBM", lgbm_summary),
        ]
    )

    # Add Trading PnL Simulation section
    md_lines.extend(
        [
            "",
            "### Trading PnL Simulation (TP=2%, SL=0.7%, Fees=0.3% round-trip)",
            "",
        ]
    )

    if "error" in probe_pnl:
        md_lines.append(f"- Trading simulation unavailable: {probe_pnl['error']}")
    else:
        # Display results for each model
        for model_name, model_key in [("Logistic Regression", "logreg"), ("LightGBM", "lgbm")]:
            model_pnl = probe_pnl.get(model_key, {})
            if "error" in model_pnl:
                md_lines.append(f"**{model_name}**: {model_pnl['error']}")
                md_lines.append("")
                continue
            
            per_threshold = model_pnl.get("per_threshold", {})
            if not per_threshold:
                md_lines.append(f"**{model_name}**: No threshold data available")
                md_lines.append("")
                continue
            
            md_lines.append(f"**{model_name}** (data range: {model_pnl.get('date_range_days', 0):.0f} days, {model_pnl.get('total_samples', 0)} samples)")
            md_lines.append("")
            md_lines.append("| Threshold | Trades | Trades/Day | Win Rate | PnL/Trade | PnL/Day | PnL/Month | Sharpe |")
            md_lines.append("|----------:|-------:|-----------:|---------:|----------:|--------:|----------:|-------:|")
            
            for threshold_key in ["60%", "70%", "80%", "90%"]:
                if threshold_key not in per_threshold:
                    continue
                t = per_threshold[threshold_key]
                md_lines.append(
                    f"| {threshold_key} | "
                    f"{t['n_trades']:,} | "
                    f"{t['trades_per_day']:.2f} | "
                    f"{t['win_rate']:.1%} | "
                    f"{t['avg_pnl_per_trade_pct']:.4f}% | "
                    f"{t['avg_pnl_per_day_pct']:.4f}% | "
                    f"{t['avg_pnl_per_month_pct']:.2f}% | "
                    f"{t['sharpe_estimate']:.2f} |"
                )
            md_lines.append("")

    md_lines.extend(
        [
            "",
            "### Per-specialist model reliability vs target (MI / R^2)",
        ]
    )

    per_model = model_reliability.get("per_model", {}) if isinstance(model_reliability, dict) else {}
    if per_model:
        for group_name, stats in per_model.items():
            md_lines.append(
                "- "
                + f"{group_name}: "
                + f"n_features={int(stats.get('n_features', 0))}, "
                + f"MI_mean={float(stats.get('mi_mean_avg', 0.0)):.4f}, "
                + f"R^2_mean={float(stats.get('r2_mean', 0.0)):.4f}, "
                + f"high_MI={int(stats.get('n_high_mi', 0))}, "
                + f"high_R^2={int(stats.get('n_high_r2', 0))}"
            )
    else:
        md_lines.append("- Per-model reliability metrics unavailable")

    md_lines.extend(
        [
            "",
            "### Per-specialist data coverage",
        ]
    )

    coverage_info = model_coverage if isinstance(model_coverage, dict) else {}
    if coverage_info:
        total_target = int(len(y))
        if total_target > 0:
            md_lines.append(f"*(Target samples: {total_target})*")

        for group_name in sorted(coverage_info.keys()):
            info = coverage_info.get(group_name, {})
            n_samples = int(info.get("n_samples", 0) or 0)
            frac = float(info.get("fraction_of_target", 0.0) or 0.0)
            start = info.get("start")
            end = info.get("end")

            line = f"- **{group_name}**: n={n_samples} ({frac:.1%} coverage)"
            if start is not None and end is not None:
                line += f", range: {start} → {end}"

                # Check for significant mismatch
                if start > y_start + pd.Timedelta(days=7):
                    line += " ⚠️ Starts late"
                if end < y_end - pd.Timedelta(days=7):
                    line += " ⚠️ Ends early"
            else:
                line += ", range unavailable"

            if frac < 0.5:
                line += " ⚠️ Low coverage (<50%)"

            md_lines.append(line)
    else:
        md_lines.append("- Per-specialist coverage unavailable")

    md_lines.extend(
        [
            "",
            "### Pairwise relationships between specialist models (MI / R^2)",
        ]
    )

    pairwise_info = model_relationships if isinstance(model_relationships, dict) else {}
    pair_list = pairwise_info.get("pairs", []) if isinstance(pairwise_info, dict) else []

    if not pair_list:
        error_msg = pairwise_info.get("error") if isinstance(pairwise_info, dict) else None
        if error_msg:
            md_lines.append(f"- Pairwise model analysis unavailable: {error_msg}")
        else:
            md_lines.append("- Pairwise model analysis unavailable")
    else:
        md_lines.append("")
        md_lines.append("| Model i | Model j | Rep feature i | Rep feature j | MI_proxy | R^2 |")
        md_lines.append("|---------|---------|---------------|---------------|---------:|----:|")
        for entry in pair_list:
            md_lines.append(
                "| "
                + f"{entry.get('model_i', '')} | "
                + f"{entry.get('model_j', '')} | "
                + f"{entry.get('rep_feature_i', '')} | "
                + f"{entry.get('rep_feature_j', '')} | "
                + f"{float(entry.get('mi_proxy', 0.0)):.4f} | "
                + f"{float(entry.get('r2', 0.0)):.4f} |"
            )

    md_lines.extend(
        [
            "",
            "### Global stability (TimeSeriesSplit AUC)",
        ]
    )

    if "error" in global_stability:
        md_lines.append(f"- Stability analysis unavailable: {global_stability['error']}")
    else:
        md_lines.append(
            f"- Mean AUC={global_stability.get('mean_auc', float('nan')):.3f}, "
            f"std={global_stability.get('std_auc', float('nan')):.3f}, "
            f"stability score={global_stability.get('stability_score', float('nan')):.3f}"
        )

    md_lines.extend(
        [
            "",
            "## Top Features by MI Proxy (CV-averaged)",
            "",
            "| Feature | MI_full | MI_mean | MI_CV | Corr | R^2 |",
            "|---------|--------:|--------:|------:|-----:|----:|",
        ]
    )

    for name, met in sorted_by_mi[:top_k]:
        md_lines.append(
            f"| {name} | {met['mi_proxy_full']:.4f} | {met['mi_mean_cv']:.4f} | "
            f"{met['mi_cv']:.3f} | {met['pearson_corr']:.3f} | {met['r2']:.4f} |"
        )

    md_lines.extend(
        [
            "",
            "## Top Features by R^2 (Univariate)",
            "",
            "| Feature | MI_full | MI_mean | MI_CV | Corr | R^2 |",
            "|---------|--------:|--------:|------:|-----:|----:|",
        ]
    )

    for name, met in sorted_by_r2[:top_k]:
        md_lines.append(
            f"| {name} | {met['mi_proxy_full']:.4f} | {met['mi_mean_cv']:.4f} | "
            f"{met['mi_cv']:.3f} | {met['pearson_corr']:.3f} | {met['r2']:.4f} |"
        )

    # Append constant/near-constant feature check
    md_lines.extend(
        [
            "",
            "## Constant / Near-Constant Feature Check",
        ]
    )
    constant_feats = []
    for col in X.columns:
        if np.std(X[col]) < 1e-9:
            val = float(X[col].mean()) if len(X) > 0 else 0.0
            constant_feats.append(f"{col} (val={val:.4f})")

    if constant_feats:
        md_lines.append(f"⚠️ Found {len(constant_feats)} constant features:")
        for f in constant_feats[:10]:
            md_lines.append(f"- {f}")
        if len(constant_feats) > 10:
            md_lines.append(f"- ... and {len(constant_feats) - 10} more")
    else:
        md_lines.append("- No constant features found (std < 1e-9).")

    # Append leakage and interaction summaries
    md_lines.extend(
        [
            "",
            "## Leakage diagnostics",
        ]
    )

    if "error" in leakage_diagnostics:
        md_lines.append(f"- Leakage detection unavailable: {leakage_diagnostics['error']}")
    else:
        susp = leakage_diagnostics.get("suspicious_features", [])
        perf = leakage_diagnostics.get("perfect_features", [])
        md_lines.append(f"- Suspicious features (|corr|>=0.95): {len(susp)}")
        md_lines.append(f"- Perfect-correlation features (|corr|>=0.99): {len(perf)}")
        if susp:
            md_lines.append("- Examples (suspicious): " + ", ".join(f"{f[0]}({f[1]:.3f})" for f in susp[:5]))
        if perf:
            md_lines.append("- Examples (perfect): " + ", ".join(f"{f[0]}({f[1]:.3f})" for f in perf[:5]))

    md_lines.extend(
        [
            "",
            "## Notable pairwise interactions (TreeSHAP)",
        ]
    )

    if "error" in interactions:
        md_lines.append(f"- Interaction analysis unavailable: {interactions['error']}")
    else:
        top_pairs = interactions.get("top_pairs", [])
        md_lines.append(
            f"- Computed on {interactions.get('n_features', 0)} features, "
            f"sample_size={interactions.get('sample_size', 0)}"
        )
        md_lines.append("")
        md_lines.append("| Feature i | Feature j | Interaction strength |")
        md_lines.append("|----------|----------|---------------------:|")
        for p in top_pairs[:20]:
            md_lines.append(
                f"| {p['feature_i']} | {p['feature_j']} | {p['interaction_strength']:.4e} |"
            )

    payload: Dict[str, Any] = {
        "summary": summary,
        "feature_metrics": feature_metrics,
        "cv_folds": int(cv_folds),
        "probe_models": probe_models,
        "probe_pnl_simulation": probe_pnl,
        "leakage_diagnostics": leakage_diagnostics,
        "stability_metrics": global_stability,
        "interactions": interactions,
        "model_reliability": model_reliability,
        "model_pairwise": model_relationships,
        "model_coverage": model_coverage,
    }

    return _export_report(
        prefix="specialist_feature_diagnostics",
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        direction=direction,
        model=model,
        payload=payload,
        markdown_lines=md_lines,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Diagnostics for specialist model outputs vs meta-label targets",
    )
    ap.add_argument("--symbol", type=str, default="ETHUSDT")
    ap.add_argument("--exchange", type=str, default="binance")
    ap.add_argument("--timeframe", type=str, default="15m")
    ap.add_argument("--direction", type=str, default="long", choices=["long", "short", "both"])
    ap.add_argument("--model", type=str, default="analyst")
    ap.add_argument("--regime-timeframe", type=str, default="1h")
    # Default target column is now direction-aware: binary_label_long for longs, binary_label_short for shorts
    # Falls back to unified binary_label if directional labels not available
    ap.add_argument("--target-col", type=str, default=None,
                    help="Target column for diagnostics. If not specified, uses direction-aware default: "
                         "binary_label_long (for long), binary_label_short (for short), or binary_label (for both)")
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument(
        "--lookback-days",
        type=float,
        default=None,
        help=(
            "Optional calendar-day lookback window. When set, diagnostics "
            "are restricted to the last N days of labeled data and "
            "specialist outputs (default: full history)."
        ),
    )
    ap.add_argument(
        "--projection-mode",
        type=str,
        choices=["canonical_scalars", "raw"],
        default="canonical_scalars",
        help=(
            "How to project specialist outputs before diagnostics. "
            "'canonical_scalars' (default) collapses each specialist to a "
            "single scalar; 'raw' uses the full specialist feature block "
            "exactly as training sees it from get_specialist_models_outputs."
        ),
    )
    ap.add_argument(
        "--disable-hmm-risk-specialist",
        dest="enable_hmm_risk_specialist",
        action="store_false",
        help=(
            "When set, disable the optional HMM risk specialist block "
            "(enable_risk_hmm_specialist=False in get_specialist_models_outputs). "
            "By default this is enabled so diagnostics and training include "
            "the HMM risk specialist."
        ),
    )
    ap.add_argument(
        "--train-specialists",
        nargs="*",
        default=None,
        help="Run training for specialist models before diagnostics. Specify list of models or leave empty for all.",
    )
    ap.add_argument(
        "--compare-targets",
        action="store_true",
        help="If set, also runs diagnostics on dense target_long/target_short for comparison."
    )

    args = ap.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    # Optional: Run training first if requested
    if args.train_specialists is not None:
        import asyncio
        asyncio.run(_run_specialist_training(
            symbol=args.symbol,
            exchange=args.exchange,
            timeframe=args.timeframe,
            direction=args.direction,
            regime_timeframe=args.regime_timeframe,
            lookback_days=args.lookback_days,
            selected_specialists=args.train_specialists,
        ))

    # Determine default target column based on direction if not explicitly provided
    if args.target_col is None:
        if args.direction == "long":
            args.target_col = "binary_label_long"
        elif args.direction == "short":
            args.target_col = "binary_label_short"
        else:
            args.target_col = "binary_label"  # Fallback for 'both' direction
        print(f"Using direction-aware default target: {args.target_col}")

    targets_to_run = [args.target_col]
    if args.compare_targets:
        # Add regression targets for comparison
        if args.direction == "long":
            if "target_long" not in targets_to_run:
                targets_to_run.append("target_long")
            # Also compare with unified binary_label for reference
            if "binary_label" not in targets_to_run:
                targets_to_run.append("binary_label")
        elif args.direction == "short":
            if "target_short" not in targets_to_run:
                targets_to_run.append("target_short")
            if "binary_label" not in targets_to_run:
                targets_to_run.append("binary_label")
        elif args.direction == "both":
            if "target_long" not in targets_to_run:
                targets_to_run.append("target_long")
            if "target_short" not in targets_to_run:
                targets_to_run.append("target_short")
            if "binary_label_long" not in targets_to_run:
                targets_to_run.append("binary_label_long")
            if "binary_label_short" not in targets_to_run:
                targets_to_run.append("binary_label_short")

    for tgt in targets_to_run:
        print(f"\n--- Running diagnostics for target: {tgt} ---")
        try:
            md_path, csv_path = run_diagnostics(
                symbol=args.symbol,
                exchange=args.exchange,
                timeframe=args.timeframe,
                direction=args.direction,
                model=args.model,
                regime_timeframe=args.regime_timeframe,
                target_col=tgt,
                cv_folds=args.cv_folds,
                lookback_days=args.lookback_days,
                enable_risk_hmm_specialist=args.enable_hmm_risk_specialist,
                projection_mode=args.projection_mode,
            )
            print(
                f"Specialist feature diagnostics for {tgt} saved to: {md_path} "
                f"and {csv_path}"
            )
        except Exception as e:
            print(f"Failed diagnostics for {tgt}: {e}")


if __name__ == "__main__":
    main()
