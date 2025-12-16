"""Shared loader for specialist model outputs (ML Risk, HMM Meso Trend, Liquidity, Breakout/Bounce).

This utility loads specialist regime outputs from versioned artifacts and
aligns them to a common training index, returning a single DataFrame with
all available specialist features.

Intended consumers:
- unified_models_training_step
- feature_generation_meta_labeling_step
- meta_labeling_hpo_experiment_step
- snr_diagnostics
- meta_gated_backtest_step
- final_parameters_optimization
"""

from typing import Any, Dict, List, Optional
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.tprint import tprint_info, tprint_warning, tprint_success
from src.utils.versioned_artifacts import VersionedArtifactStore


def _standardize_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame has a DatetimeIndex based on 'timestamp' or its index.

    This helper is intentionally conservative: it avoids timezone conversions
    and relies on `pandas.to_datetime(..., errors="coerce")`. If timestamps
    cannot be parsed, it raises so callers can skip that specialist block.
    """
    if df is None or getattr(df, "empty", True):
        return df

    # Prefer explicit timestamp column when available
    if "timestamp" in df.columns:
        df = df.copy()
        ts = pd.to_datetime(df["timestamp"], errors="coerce")

        valid_mask = ~ts.isna()
        if not bool(valid_mask.all()):
            df = df.loc[valid_mask].copy()
            ts = ts[valid_mask]

        df = df.drop(columns=["timestamp"])
        df.index = ts
        df.index.name = "timestamp"

    elif not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        idx = pd.to_datetime(df.index, errors="coerce")
        valid_mask = ~idx.isna()
        if not bool(valid_mask.any()):
            raise ValueError("DataFrame index is not a DatetimeIndex and cannot be converted")
        if not bool(valid_mask.all()):
            df = df.loc[valid_mask].copy()
            idx = idx[valid_mask]
        df.index = idx

    # If we already had a DatetimeIndex, ensure it is monotonic and
    # de-duplicated for downstream reindex(..., method="ffill") calls.
    if isinstance(df.index, pd.DatetimeIndex):
        if not (
            df.index.is_monotonic_increasing or df.index.is_monotonic_decreasing
        ):
            df = df.sort_index()

        if df.index.has_duplicates:
            # Keep the last occurrence for each timestamp so that we retain
            # the most recent specialist signal before alignment.
            df = df[~df.index.duplicated(keep="last")]

    return df


def _project_specialists_to_canonical_scalars(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse specialist outputs to canonical scalar features per specialist.

    This is shared between training and diagnostics when
    `use_canonical_specialist_scalars=True` is set in the config passed to
    `get_specialist_models_outputs`.

    Conventions:
    - Risk: single scalar `risk_score` in [0, 1]
    - Liquidity: keep regime probabilities as-is
    - Breakout: keep directional S/R scalars and success-probability signals
    - Path/macro: single scalar per specialist
    - SMC: single scalar `smc_predicted`
    - Mean reversion: single student score (`mr_probability` or `mr_raw_score`)
    """

    X = df.copy()
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
        if risk_drop:
            X = X.drop(columns=risk_drop, errors="ignore")
            cols = list(X.columns)

    # ------------------------------------------------------------------
    # Breakout/Bounce: keep S/R scalars (+ success prob/high-confidence flag)
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
        if path_drop:
            X = X.drop(columns=path_drop, errors="ignore")
            cols = list(X.columns)

    # ------------------------------------------------------------------
    # SMC: keep a single scalar (smc_predicted) and drop auxiliary columns
    # ------------------------------------------------------------------
    smc_cols = [c for c in cols if c.startswith("smc_")]
    smc_keep = set()
    if "smc_predicted" in X.columns:
        smc_keep.add("smc_predicted")

    smc_drop = [c for c in smc_cols if c not in smc_keep]
    if smc_drop:
        X = X.drop(columns=smc_drop, errors="ignore")
        cols = list(X.columns)

    # ------------------------------------------------------------------
    # Mean reversion: keep a single XGB score, preferring dense per-bar
    # probabilities when available. Priority:
    #   1) mr_probability_dense (dense XGB-based scalar per bar)
    #   2) mr_probability       (OOF-only calibrated probabilities)
    #   3) mr_raw_score         (uncalibrated OOF score)
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


def get_specialist_models_outputs(
    *,
    artifact_router: Any,
    training_index: pd.DatetimeIndex,
    config: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
    strict: bool = True,
) -> Optional[pd.DataFrame]:
    """Load specialist model outputs (ML Risk, Liquidity, Breakout/Bounce) and
    align them to the provided training_index.

    All returned features are aligned to training_index via forward-fill.

    Args:
        artifact_router: Artifact router instance with a `.load(...)` method.
        training_index: Target DatetimeIndex for alignment (e.g., 15m index).
        config: Pipeline configuration dict (must contain symbol/exchange/timeframe).
        logger: Optional logger for debug/warning messages.
        strict: If True, raise ValueError when no specialist sources are found.

    Returns:
        A single DataFrame with all specialist features aligned to
        training_index, or None if no sources found and strict=False.
    """

    def log_info(msg: str) -> None:
        tprint_info(msg)
        if logger is not None:
            logger.info(msg)

    def log_warning(msg: str) -> None:
        tprint_warning(msg)
        if logger is not None:
            logger.warning(msg)

    def log_success(msg: str) -> None:
        tprint_success(msg)
        if logger is not None:
            logger.info(msg)

    symbol = config.get("symbol", "ETHUSDT")
    exchange = config.get("exchange", "binance")
    direction = config.get("direction", "long")
    base_timeframe = str(config.get("timeframe", "15m"))
    # Force regime_timeframe to match base_timeframe (15m) to ensure we load
    # the correct artifacts, ignoring any 1h overrides in config.
    regime_timeframe = base_timeframe

    # Ensure training_index is a DatetimeIndex (naive) for reindexing.
    #
    # IMPORTANT: Converting a plain RangeIndex (0..N) via pd.to_datetime will
    # silently produce 1970-01-01-based timestamps at nanosecond offsets,
    # which breaks temporal alignment and leads to all-NaN specialist blocks.
    # Instead, require a real DatetimeIndex; if we only have a RangeIndex or
    # cannot obtain valid timestamps, skip specialist outputs (or raise in
    # strict mode).
    if not isinstance(training_index, pd.DatetimeIndex):
        if isinstance(training_index, pd.RangeIndex):
            log_warning(
                "⚠️ Training index is a RangeIndex; specialist outputs require a "
                "real DatetimeIndex for alignment. Skipping specialist blocks."
            )
            if strict:
                raise ValueError(
                    "Training index must be a DatetimeIndex to align specialist outputs"
                )
            return None

        training_index = pd.to_datetime(training_index, errors="coerce")

    # If conversion failed or produced no valid timestamps, bail out early.
    if not isinstance(training_index, pd.DatetimeIndex) or len(training_index) == 0:
        log_warning(
            "⚠️ Training index could not be converted to a valid DatetimeIndex; "
            "skipping specialist outputs."
        )
        if strict:
            raise ValueError(
                "Training index must be a valid DatetimeIndex for specialist outputs"
            )
        return None

    # Basic diagnostics on the target index used for alignment
    log_info(
        "🎯 Training index range: %s → %s (n=%d)"
        % (training_index.min(), training_index.max(), len(training_index))
    )

    blocks: List[pd.DataFrame] = []

    # ------------------------------------------------------------------
    # 1) ML Risk regimes – prefer 15m probabilities if available
    # ------------------------------------------------------------------
    try:
        log_info("=" * 80)
        log_info("🌟 LOADING SPECIALIST: ML RISK REGIME OUTPUTS")
        log_info("=" * 80)

        # Preferred: upsampled probabilities at the base timeframe
        risk_probs_context = {
            "symbol": symbol,
            "exchange": exchange,
            "timeframe": base_timeframe,
            "direction": direction,
            "model": "regime_risk",
            "step_name": "ml_risk_regime_step",
        }
        risk_probs_name = f"ml_risk_regime_probabilities_{base_timeframe}"
        log_info(f"Attempting to load ML Risk probabilities: {risk_probs_name}")
        risk_probs = artifact_router.load(
            artifact_name=risk_probs_name,
            artifact_type="data",
            data_category="features",
            context=risk_probs_context,
        )

        if risk_probs is not None and not getattr(risk_probs, "empty", True):
            if not isinstance(risk_probs, pd.DataFrame):
                risk_probs = pd.DataFrame(risk_probs)
            risk_probs = _standardize_index(risk_probs)

            if isinstance(risk_probs.index, pd.DatetimeIndex) and len(risk_probs.index) > 0:
                log_info(
                    "📈 ML Risk (%s) index range: %s → %s (n=%d)"
                    % (
                        risk_probs_name,
                        risk_probs.index.min(),
                        risk_probs.index.max(),
                        len(risk_probs.index),
                    )
                )

            # Select regime probability columns (and label if present)
            risk_cols = [
                c
                for c in risk_probs.columns
                if c.startswith("risk_regime_") and "prob" in c
            ]
            if "risk_regime" in risk_probs.columns:
                risk_cols.append("risk_regime")

            if risk_cols:
                before_block = risk_probs[risk_cols].copy()
                nnz_before = int(before_block.notna().sum().sum())
                # Pre-reindex NaN diagnostics on the native specialist index
                try:
                    log_info("🔍 [ML RISK] Pre-reindex NaN coverage by column:")
                    for c in before_block.columns:
                        total = len(before_block)
                        nnz_col = int(before_block[c].notna().sum())
                        ratio = nnz_col / float(total) if total > 0 else 0.0
                        log_info(f"   - {c}: non-null={nnz_col}, ratio={ratio:.3f}")
                except Exception as diag_exc:
                    log_warning(f"⚠️ Failed to log ML Risk pre-reindex NaN coverage: {diag_exc}")

                block = before_block.shift(1).fillna(method="ffill")
                block = block.reindex(training_index, method="ffill")
                nnz_after = int(block.notna().sum().sum())
                blocks.append(block)
                log_success(
                    f"✅ Added ML Risk specialist block from '{risk_probs_name}': "
                    f"shape={block.shape}, non_null_before={nnz_before}, "
                    f"non_null_after={nnz_after}"
                )
                if nnz_after == 0:
                    log_warning(
                        "⚠️ ML Risk block aligned to training_index is all-NaN. "
                        "Check risk artifacts and label index overlap."
                    )

            # Additionally, when the optional HMM risk specialist is disabled,
            # expose the normalized scalar risk_score from the primary ML Risk
            # training artifact (ml_risk_training_data_15m). This ensures that
            # both training and diagnostics can use the same scalar output from
            # ml_risk_regime_step without requiring the separate HMM block.
            if not config.get("enable_risk_hmm_specialist", True):
                try:
                    risk_training_ctx = {
                        "symbol": symbol,
                        "exchange": exchange,
                        "timeframe": regime_timeframe,
                        "direction": direction,
                        "model": "regime_risk",
                        "step_name": "ml_risk_regime_step",
                    }

                    risk_training_for_score = artifact_router.load(
                        artifact_name="ml_risk_training_data_15m",
                        artifact_type="data",
                        data_category="features",
                        context=risk_training_ctx,
                    )

                    if risk_training_for_score is not None and not getattr(
                        risk_training_for_score, "empty", True
                    ):
                        if not isinstance(risk_training_for_score, pd.DataFrame):
                            risk_training_for_score = pd.DataFrame(risk_training_for_score)
                        risk_training_for_score = _standardize_index(risk_training_for_score)

                        if "risk_score" in risk_training_for_score.columns:
                            # Avoid duplicating risk_score if it is already
                            # present in a previously-added block.
                            already_has_risk_score = any(
                                "risk_score" in getattr(b, "columns", []) for b in blocks
                            )
                            if not already_has_risk_score:
                                score_block = risk_training_for_score[["risk_score"]].copy()
                                nnz_before_score = int(score_block.notna().sum().sum())
                                score_block = score_block.shift(1).fillna(method="ffill")
                                score_block = score_block.reindex(training_index, method="ffill")
                                nnz_after_score = int(score_block.notna().sum().sum())
                                blocks.append(score_block)
                                log_success(
                                    "✅ Added ML Risk scalar 'risk_score' block from 'ml_risk_training_data_15m': "
                                    f"shape={score_block.shape}, non_null_before={nnz_before_score}, "
                                    f"non_null_after={nnz_after_score}"
                                )
                except Exception as e_score:
                    log_warning(
                        f"⚠️ Failed to load ML Risk scalar 'risk_score' from training data: {e_score}"
                    )
        else:
            # Fallback: 15m training data on the regime timeframe
            risk_context = {
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": regime_timeframe,
                "direction": direction,
                "model": "regime_risk",
                "step_name": "ml_risk_regime_step",
            }
            risk_training = artifact_router.load(
                artifact_name="ml_risk_training_data_15m",
                artifact_type="data",
                data_category="features",
                context=risk_context,
            )

            if risk_training is not None and not getattr(risk_training, "empty", True):
                if not isinstance(risk_training, pd.DataFrame):
                    risk_training = pd.DataFrame(risk_training)
                risk_training = _standardize_index(risk_training)

                if isinstance(risk_training.index, pd.DatetimeIndex) and len(risk_training.index) > 0:
                    log_info(
                        "📈 ML Risk (ml_risk_training_data_15m) index range: %s → %s (n=%d)"
                        % (
                            risk_training.index.min(),
                            risk_training.index.max(),
                            len(risk_training.index),
                        )
                    )

                risk_cols = [
                    c
                    for c in risk_training.columns
                    if c.startswith("risk_regime")
                    or c.startswith("risk_pred_")
                    or c == "risk_score"
                ]
                if risk_cols:
                    before_block = risk_training[risk_cols].copy()
                    nnz_before = int(before_block.notna().sum().sum())
                    block = before_block.shift(1).fillna(method="ffill")
                    block = block.reindex(training_index, method="ffill")
                    nnz_after = int(block.notna().sum().sum())
                    blocks.append(block)
                    log_success(
                        "✅ Added ML Risk specialist block from 'ml_risk_training_data_15m': "
                        f"shape={block.shape}, non_null_before={nnz_before}, "
                        f"non_null_after={nnz_after}"
                    )
                    if nnz_after == 0:
                        log_warning(
                            "⚠️ ML Risk 1h training block aligned to training_index is all-NaN. "
                            "Check risk_training timestamps and values."
                        )

            if config.get("enable_risk_hmm_specialist", True):
                try:
                    risk_hmm_ctx = {
                        "symbol": symbol,
                        "exchange": exchange,
                        "timeframe": base_timeframe,
                        "direction": direction,
                        "model": "regime_risk_hmm",
                        "step_name": "ml_risk_regime_step_hmm",
                    }

                    risk_hmm_artifact = f"ml_risk_hmm_training_data_{base_timeframe}"
                    risk_hmm_training = artifact_router.load(
                        artifact_name=risk_hmm_artifact,
                        artifact_type="data",
                        data_category="features",
                        context=risk_hmm_ctx,
                    )

                    if risk_hmm_training is not None and not getattr(
                        risk_hmm_training, "empty", True
                    ):
                        if not isinstance(risk_hmm_training, pd.DataFrame):
                            risk_hmm_training = pd.DataFrame(risk_hmm_training)
                        risk_hmm_training = _standardize_index(risk_hmm_training)

                        hmm_cols = [
                            c
                            for c in ("risk_score", "risk_regime")
                            if c in risk_hmm_training.columns
                        ]

                        already_has_risk_score = any(
                            "risk_score" in getattr(b, "columns", []) for b in blocks
                        )

                        if hmm_cols and not already_has_risk_score:
                            before_block = risk_hmm_training[hmm_cols].copy()
                            nnz_before = int(before_block.notna().sum().sum())
                            block = before_block.shift(1).fillna(method="ffill")
                            block = block.reindex(training_index, method="ffill")
                            nnz_after = int(block.notna().sum().sum())
                            blocks.append(block)
                            log_success(
                                "✅ Added ML Risk HMM specialist block from '%s': shape=%s, non_null_before=%d, non_null_after=%d"
                                % (risk_hmm_artifact, block.shape, nnz_before, nnz_after)
                            )
                except Exception as hmm_exc:
                    log_warning(
                        f"⚠️ Failed to load ML Risk HMM specialist outputs: {hmm_exc}"
                    )
    except Exception as e:
        log_warning(f"⚠️ Failed to load ML Risk specialist outputs: {e}")

        if config.get("enable_risk_hmm_specialist", True):
            try:
                risk_hmm_ctx = {
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": base_timeframe,
                    "direction": direction,
                    "model": "regime_risk_hmm",
                    "step_name": "ml_risk_regime_step_hmm",
                }

                risk_hmm_artifact = f"ml_risk_hmm_training_data_{base_timeframe}"
                risk_hmm_training = artifact_router.load(
                    artifact_name=risk_hmm_artifact,
                    artifact_type="data",
                    data_category="features",
                    context=risk_hmm_ctx,
                )

                if risk_hmm_training is not None and not getattr(
                    risk_hmm_training, "empty", True
                ):
                    if not isinstance(risk_hmm_training, pd.DataFrame):
                        risk_hmm_training = pd.DataFrame(risk_hmm_training)
                    risk_hmm_training = _standardize_index(risk_hmm_training)

                    hmm_cols = [
                        c
                        for c in ("risk_score", "risk_regime")
                        if c in risk_hmm_training.columns
                    ]
                    already_has_risk_score = any(
                        "risk_score" in getattr(b, "columns", []) for b in blocks
                    )
                    if hmm_cols and not already_has_risk_score:
                        before_block = risk_hmm_training[hmm_cols].copy()
                        nnz_before = int(before_block.notna().sum().sum())
                        block = before_block.shift(1).fillna(method="ffill")
                        block = block.reindex(training_index, method="ffill")
                        nnz_after = int(block.notna().sum().sum())
                        blocks.append(block)
                        log_success(
                            "✅ Added ML Risk HMM specialist block from '%s': shape=%s, non_null_before=%d, non_null_after=%d"
                            % (risk_hmm_artifact, block.shape, nnz_before, nnz_after)
                        )
            except Exception as hmm_exc:
                log_warning(f"⚠️ Failed to load ML Risk HMM specialist outputs: {hmm_exc}")

    # ------------------------------------------------------------------
    # 3) Liquidity regimes – 15m probabilities
    # ------------------------------------------------------------------
    try:
        log_info("=" * 80)
        log_info("💧 LOADING SPECIALIST: LIQUIDITY REGIME OUTPUTS")
        log_info("=" * 80)

        liquidity_context = {
            "symbol": symbol,
            "exchange": exchange,
            "timeframe": base_timeframe,
            "direction": direction,
            "model": "liquidity_regime",
            "step_name": "ml_liquidity_regime_step",
        }

        liquidity_artifact_name = f"ml_liquidity_regime_probs_{base_timeframe}"
        log_info(f"Attempting to load Liquidity probabilities: {liquidity_artifact_name}")
        liquidity_probs = artifact_router.load(
            artifact_name=liquidity_artifact_name,
            artifact_type="data",
            data_category="features",
            context=liquidity_context,
        )

        if liquidity_probs is not None and not getattr(liquidity_probs, "empty", True):
            if not isinstance(liquidity_probs, pd.DataFrame):
                liquidity_probs = pd.DataFrame(liquidity_probs)
            liquidity_probs = _standardize_index(liquidity_probs)

            if isinstance(liquidity_probs.index, pd.DatetimeIndex) and len(liquidity_probs.index) > 0:
                log_info(
                    "📈 Liquidity (ml_liquidity_regime_probs_15m) index range: %s → %s (n=%d)"
                    % (
                        liquidity_probs.index.min(),
                        liquidity_probs.index.max(),
                        len(liquidity_probs.index),
                    )
                )

            prob_cols = [
                c
                for c in liquidity_probs.columns
                if c.startswith("liquidity_regime_") and c.endswith("_prob")
            ]

            if prob_cols:
                before_block = liquidity_probs[prob_cols].copy()
                nnz_before = int(before_block.notna().sum().sum())
                try:
                    log_info("🔍 [LIQUIDITY] Pre-reindex NaN coverage by column:")
                    for c in before_block.columns:
                        total = len(before_block)
                        nnz_col = int(before_block[c].notna().sum())
                        ratio = nnz_col / float(total) if total > 0 else 0.0
                        log_info(f"   - {c}: non-null={nnz_col}, ratio={ratio:.3f}")
                except Exception as diag_exc:
                    log_warning(f"⚠️ Failed to log Liquidity pre-reindex NaN coverage: {diag_exc}")

                block = before_block.reindex(training_index, method="ffill")
                nnz_after = int(block.notna().sum().sum())

                try:
                    rows_nonnull = int(block.notna().any(axis=1).sum())
                    total_rows = int(len(training_index))
                    coverage_ratio = (
                        rows_nonnull / float(total_rows) if total_rows > 0 else 0.0
                    )
                except Exception:
                    rows_nonnull = 0
                    coverage_ratio = 0.0

                if coverage_ratio < 0.9 and len(training_index) > 0:
                    try:
                        store_name = f"{symbol}_{exchange}_{base_timeframe}_{direction}_liquidity_regime"
                        store_path = Path("versioned_artifacts") / store_name
                        store = VersionedArtifactStore(store_path)

                        best_version = None
                        best_rows = 0
                        for v in store.list_versions():
                            info = store.get_version_info(v)
                            if info.get("artifact_name") != liquidity_artifact_name:
                                continue
                            if str(info.get("timeframe")) != base_timeframe:
                                continue
                            src_tf = str(info.get("source_regime_timeframe", base_timeframe))
                            if src_tf != base_timeframe:
                                continue
                            num_rows = int(info.get("num_rows", 0) or 0)
                            if num_rows > best_rows:
                                best_rows = num_rows
                                best_version = v

                        if best_version is not None:
                            log_info(
                                "🔁 [LIQUIDITY] Low coverage detected ("
                                f"{coverage_ratio:.3f}); reloading full 15m probabilities "
                                f"from version '{best_version}'."
                            )
                            view = store.get_view(best_version)
                            full_probs = view.materialize()
                            if not isinstance(full_probs, pd.DataFrame):
                                full_probs = pd.DataFrame(full_probs)
                            full_probs = _standardize_index(full_probs)

                            # Recompute probability columns from the reloaded
                            # DataFrame to handle cases where the column set
                            # differs from the original pickle (e.g., a
                            # different number of liquidity_regime_*_prob
                            # columns). This ensures we always use the
                            # columns that actually exist in the dense,
                            # long-window VersionedArtifactStore rather than
                            # failing with a KeyError or keeping the
                            # truncated block.
                            prob_cols_full = [
                                c
                                for c in full_probs.columns
                                if c.startswith("liquidity_regime_") and c.endswith("_prob")
                            ]

                            if prob_cols_full:
                                before_block = full_probs[prob_cols_full].copy()
                                block = before_block.reindex(training_index, method="ffill")
                                nnz_after = int(block.notna().sum().sum())
                                rows_nonnull = int(block.notna().any(axis=1).sum())
                                try:
                                    log_info(
                                        "🔁 [LIQUIDITY] Coverage after reload (VersionedArtifactStore): "
                                        f"rows_nonnull={rows_nonnull}/{len(training_index)}"
                                    )
                                except Exception:
                                    pass
                            else:
                                log_warning(
                                    "⚠️ [LIQUIDITY] Reloaded full_probs from VersionedArtifactStore "
                                    "contains no liquidity_regime_*_prob columns; keeping original block."
                                )
                    except Exception as reload_exc:
                        log_warning(
                            f"⚠️ [LIQUIDITY] Coverage repair via VersionedArtifactStore "
                            f"failed: {reload_exc}"
                        )

                blocks.append(block)
                log_success(
                    "✅ Added Liquidity specialist block from 'ml_liquidity_regime_probs_15m': "
                    f"shape={block.shape}, non_null_before={nnz_before}, "
                    f"non_null_after={nnz_after}"
                )
                if nnz_after == 0:
                    log_warning(
                        "⚠️ Liquidity block aligned to training_index is all-NaN. "
                        "Check liquidity_probs timestamps and values."
                    )
    except Exception as e:
        log_warning(f" Failed to load Liquidity specialist outputs: {e}")

    # ------------------------------------------------------------------
    # 4) Breakout/Bounce regimes – 1h probabilities + edge scores
    # ------------------------------------------------------------------
    try:
        if config.get("enable_breakout_specialist", True):
            log_info("=" * 80)
            log_info(" LOADING SPECIALIST: BREAKOUT/BOUNCE REGIME OUTPUTS")
            log_info("=" * 80)

            breakout_context = {
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": base_timeframe,
                "direction": direction,
                "model": "breakout_bounce",
                "step_name": "ml_breakout_bounce_regime_step",
            }

            breakout_artifact_name = f"ml_breakout_bounce_training_data_{base_timeframe}"
            log_info(f"Attempting to load Breakout/Bounce training data: {breakout_artifact_name}")
            breakout_training = artifact_router.load(
                artifact_name=breakout_artifact_name,
                artifact_type="data",
                data_category="features",
                context=breakout_context,
            )

            if breakout_training is not None and not getattr(breakout_training, "empty", True):
                if not isinstance(breakout_training, pd.DataFrame):
                    breakout_training = pd.DataFrame(breakout_training)
                breakout_training = _standardize_index(breakout_training)

                if isinstance(breakout_training.index, pd.DatetimeIndex) and len(breakout_training.index) > 0:
                    log_info(
                        " Breakout/Bounce (ml_breakout_bounce_training_data_15m) index range: %s → %s (n=%d)"
                        % (
                            breakout_training.index.min(),
                            breakout_training.index.max(),
                            len(breakout_training.index),
                        )
                    )

                # Select a compact but expressive set of breakout features, including
                # regime probabilities, directional scalars, edge scores, side
                # indicators, and calibrated success-probability gating signals.
                breakout_candidate_cols = [
                    # Core regime probabilities
                    "breakout_regime_0_prob",
                    "breakout_regime_1_prob",
                    "breakout_regime_2_prob",
                    # Directional scalars (new unified scalar output)
                    "resistance_scalar",
                    "breakout_scalar_resistance",
                    "support_scalar",
                    "breakout_scalar_support",
                    # Edge scores
                    "breakout_long_edge_score",
                    "breakout_short_edge_score",
                    # Side indicators
                    "is_resistance",
                    "is_support",
                    # Calibrated success probability and high-confidence flag
                    "breakout_success_prob",
                    "breakout_high_conf_signal",
                ]

                breakout_cols = [
                    c
                    for c in breakout_candidate_cols
                    if c in breakout_training.columns
                ]

                if breakout_cols:
                    before_block = breakout_training[breakout_cols].copy()

                    # Normalize legacy breakout scalar aliases to canonical names so
                    # downstream consumers only see support_scalar / resistance_scalar.
                    if "breakout_scalar_support" in before_block.columns and "support_scalar" not in before_block.columns:
                        before_block["support_scalar"] = before_block["breakout_scalar_support"]
                    if "breakout_scalar_resistance" in before_block.columns and "resistance_scalar" not in before_block.columns:
                        before_block["resistance_scalar"] = before_block["breakout_scalar_resistance"]

                    alias_drop = [
                        c
                        for c in ("breakout_scalar_support", "breakout_scalar_resistance")
                        if c in before_block.columns
                    ]
                    if alias_drop:
                        before_block = before_block.drop(columns=alias_drop)

                    # Guard against using stale breakout artifacts whose index does not
                    # overlap the training_index. In that case, reindex+ffill would
                    # produce a constant block, which is misleading. Instead, skip
                    # the specialist block and emit a clear warning.
                    if isinstance(before_block.index, pd.DatetimeIndex):
                        overlap = before_block.index.intersection(training_index)
                        if overlap.empty:
                            log_warning(
                                " Breakout/Bounce specialist index has no overlap with "
                                "training_index; skipping breakout block. Check timeframe "
                                "and artifact recency."
                            )
                        else:
                            nnz_before = int(before_block.notna().sum().sum())
                            # Pre-reindex NaN diagnostics
                            try:
                                log_info(" [BREAKOUT] Pre-reindex NaN coverage by column:")
                                for c in before_block.columns:
                                    total = len(before_block)
                                    nnz_col = int(before_block[c].notna().sum())
                                    ratio = nnz_col / float(total) if total > 0 else 0.0
                                    log_info(f"   - {c}: non-null={nnz_col}, ratio={ratio:.3f}")
                            except Exception as diag_exc:
                                log_warning(f" Failed to log Breakout pre-reindex NaN coverage: {diag_exc}")

                            block = before_block.shift(1).fillna(method="ffill")
                            block = block.reindex(training_index, method="ffill")
                            nnz_after = int(block.notna().sum().sum())
                            blocks.append(block)
                            log_success(
                                " Added Breakout/Bounce specialist block from 'ml_breakout_bounce_training_data_15m': "
                                f"shape={block.shape}, non_null_before={nnz_before}, "
                                f"non_null_after={nnz_after}"
                            )
                            if nnz_after == 0:
                                log_warning(
                                    " Breakout/Bounce block aligned to training_index is all-NaN. "
                                    "Check breakout_training timestamps and values."
                                )
        else:
            log_info(" Skipping Breakout/Bounce specialist outputs (enable_breakout_specialist=False)")
    except Exception as e:
        log_warning(f" Failed to load Breakout/Bounce specialist outputs: {e}")

        if base_timeframe != "1h" and config.get("enable_breakout_specialist", True):
            try:
                breakout_context = {
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": "1h",
                    "direction": direction,
                    "model": "breakout_bounce",
                    "step_name": "ml_breakout_bounce_regime_step",
                }

                breakout_artifact_name = "ml_breakout_bounce_training_data_1h"
                breakout_training = artifact_router.load(
                    artifact_name=breakout_artifact_name,
                    artifact_type="data",
                    data_category="features",
                    context=breakout_context,
                )

                if breakout_training is not None and not getattr(
                    breakout_training, "empty", True
                ):
                    if not isinstance(breakout_training, pd.DataFrame):
                        breakout_training = pd.DataFrame(breakout_training)
                    breakout_training = _standardize_index(breakout_training)

                    breakout_candidate_cols = [
                        "breakout_regime_0_prob",
                        "breakout_regime_1_prob",
                        "breakout_regime_2_prob",
                        "resistance_scalar",
                        "breakout_scalar_resistance",
                        "support_scalar",
                        "breakout_scalar_support",
                        "breakout_long_edge_score",
                        "breakout_short_edge_score",
                        "is_resistance",
                        "is_support",
                        "breakout_success_prob",
                        "breakout_high_conf_signal",
                    ]
                    breakout_cols = [
                        c
                        for c in breakout_candidate_cols
                        if c in breakout_training.columns
                    ]

                    if breakout_cols:
                        before_block = breakout_training[breakout_cols].copy()

                        if "breakout_scalar_support" in before_block.columns and "support_scalar" not in before_block.columns:
                            before_block["support_scalar"] = before_block["breakout_scalar_support"]
                        if "breakout_scalar_resistance" in before_block.columns and "resistance_scalar" not in before_block.columns:
                            before_block["resistance_scalar"] = before_block["breakout_scalar_resistance"]

                        alias_drop = [
                            c
                            for c in ("breakout_scalar_support", "breakout_scalar_resistance")
                            if c in before_block.columns
                        ]
                        if alias_drop:
                            before_block = before_block.drop(columns=alias_drop)

                        overlap = before_block.index.intersection(training_index)
                        if not overlap.empty:
                            nnz_before = int(before_block.notna().sum().sum())
                            block = before_block.shift(1).fillna(method="ffill")
                            block = block.reindex(training_index, method="ffill")
                            nnz_after = int(block.notna().sum().sum())
                            blocks.append(block)
                            log_success(
                                " Added Breakout/Bounce specialist block from 'ml_breakout_bounce_training_data_1h': "
                                f"shape={block.shape}, non_null_before={nnz_before}, non_null_after={nnz_after}"
                            )
            except Exception as breakout_fallback_exc:
                log_warning(
                    f" Failed to load Breakout/Bounce specialist outputs (1h fallback): {breakout_fallback_exc}"
                )

    # ------------------------------------------------------------------
    # 5) Path regimes – HMM Path specialist (optional)
    # ------------------------------------------------------------------
    try:
        log_info("=" * 80)
        log_info(" LOADING SPECIALIST: PATH REGIME OUTPUTS")
        log_info("=" * 80)

        path_context = {
            "symbol": symbol,
            "exchange": exchange,
            "timeframe": regime_timeframe,
            "direction": direction,
            "model": "regime_path",
            "step_name": "ml_path_regime_step",
        }

        # Reuse the standardized training artifact produced by MLPathRegimeStep.
        path_artifact_name = f"ml_path_training_data_{base_timeframe}"
        log_info(f"Attempting to load Path training data: {path_artifact_name}")
        path_training = artifact_router.load(
            artifact_name=path_artifact_name,
            artifact_type="data",
            data_category="features",
            context=path_context,
        )

        if path_training is not None and not getattr(path_training, "empty", True):
            if not isinstance(path_training, pd.DataFrame):
                path_training = pd.DataFrame(path_training)
            path_training = _standardize_index(path_training)

            if isinstance(path_training.index, pd.DatetimeIndex) and len(path_training.index) > 0:
                log_info(
                    " Path (ml_path_training_data_15m under ml_path_regime_step) index range: %s → %s (n=%d)"
                    % (
                        path_training.index.min(),
                        path_training.index.max(),
                        len(path_training.index),
                    )
                )

            # Select path-specific features plus a dedicated path_regime label
            # derived from the stored risk_regime column (if present, otherwise trust path_regime).
            path_cols: List[str] = [
                c for c in path_training.columns if c.startswith("path_")
            ]

            # If the artifact still uses 'risk_regime' as the label column, alias it to 'path_regime'
            if "risk_regime" in path_training.columns and "path_regime" not in path_training.columns:
                path_training = path_training.copy()
                path_training["path_regime"] = path_training["risk_regime"]
                path_cols.append("path_regime")
            elif "path_regime" in path_training.columns:
                path_cols.append("path_regime")

            if path_cols:
                before_block = path_training[path_cols].copy()
                nnz_before = int(before_block.notna().sum().sum())
                # Align to training_index; use the same shift+ffill convention
                # as ML Risk so that regimes are lagged by one bar.
                block = before_block.shift(1).fillna(method="ffill")
                block = block.reindex(training_index, method="ffill")
                nnz_after = int(block.notna().sum().sum())
                blocks.append(block)
                log_success(
                    " Added Path specialist block from 'ml_path_training_data_15m' (ml_path_regime_step): "
                    f"shape={block.shape}, non_null_before={nnz_before}, "
                    f"non_null_after={nnz_after}"
                )
                if nnz_after == 0:
                    log_warning(
                        " Path block aligned to training_index is all-NaN. "
                        "Check path_training timestamps and values."
                    )
    except Exception as e:
        log_warning(f" Failed to load Path specialist outputs: {e}")

        try:
            path_probs = artifact_router.load(
                artifact_name="ml_path_regime_probabilities_15m",
                artifact_type="data",
                data_category="features",
                context=None,
            )

            if path_probs is not None and not getattr(path_probs, "empty", True):
                if not isinstance(path_probs, pd.DataFrame):
                    path_probs = pd.DataFrame(path_probs)
                path_probs = _standardize_index(path_probs)

                cols = []
                if "risk_regime" in path_probs.columns:
                    cols.append("risk_regime")
                if "risk_regime_directional" in path_probs.columns:
                    cols.append("risk_regime_directional")

                if cols:
                    before_block = path_probs[cols].copy()
                    before_block = before_block.rename(
                        columns={
                            "risk_regime": "path_regime",
                            "risk_regime_directional": "path_regime_directional",
                        }
                    )
                    nnz_before = int(before_block.notna().sum().sum())
                    block = before_block.shift(1).fillna(method="ffill")
                    block = block.reindex(training_index, method="ffill")
                    nnz_after = int(block.notna().sum().sum())
                    blocks.append(block)
                    log_success(
                        " Added Path specialist block from 'ml_path_regime_probabilities_15m': "
                        f"shape={block.shape}, non_null_before={nnz_before}, non_null_after={nnz_after}"
                    )
        except Exception as path_pickle_exc:
            log_warning(f" Failed to load Path specialist outputs (pickle fallback): {path_pickle_exc}")

    # ------------------------------------------------------------------
    # 6) Macro Trend specialist – macro regime alpha signal from xgb_macro_regime
    # ------------------------------------------------------------------
    try:
        if config.get("enable_macro_trend_specialist", True):
            log_info("=" * 80)
            log_info(" LOADING SPECIALIST: MACRO TREND (XGB) OUTPUTS")
            log_info("=" * 80)

            macro_context = {
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": regime_timeframe,
                "direction": direction,
                # Updated model namespace to match the current xgb_macro_regime
                # step, ensuring we load artifacts from the correct
                # ETHUSDT_binance_15m_long_regime_macro_trend store.
                "model": "regime_macro_trend",
                "step_name": "xgb_macro_regime",
            }

            macro_artifact_name = f"hmm_macro_trend_training_data_{base_timeframe}"
            log_info(f"Attempting to load Macro Trend data: {macro_artifact_name}")
            macro_training = artifact_router.load(
                artifact_name=macro_artifact_name,
                artifact_type="data",
                data_category="features",
                context=macro_context,
            )

            if macro_training is not None and not getattr(macro_training, "empty", True):
                if not isinstance(macro_training, pd.DataFrame):
                    macro_training = pd.DataFrame(macro_training)
                macro_training = _standardize_index(macro_training)

                if (
                    isinstance(macro_training.index, pd.DatetimeIndex)
                    and len(macro_training.index) > 0
                ):
                    log_info(
                        " Macro Trend (hmm_macro_trend_training_data_15m) index range: %s → %s (n=%d)"
                        % (
                            macro_training.index.min(),
                            macro_training.index.max(),
                            len(macro_training.index),
                        )
                    )

                # Prefer the canonical 0-1 continuous macro alpha score; fall back to
                # calibrated expectation-style columns if necessary.
                score_col: Optional[str] = None
                for c in ("macro_trend_score_continuous", "macro_trend_expectation_ema_01", "macro_trend_expectation_raw_01"):
                    if c in macro_training.columns:
                        score_col = c
                        break

                if score_col is not None:
                    before_block = macro_training[[score_col]].copy()
                    # Expose the canonical 0-1 macro trend scalar under the
                    # established name used in analyst training and diagnostics.
                    # This ensures downstream users see a single
                    # `macro_trend_score_continuous` feature backed by the new
                    # XGB macro regime model rather than any legacy macro outputs.
                    before_block = before_block.rename(
                        columns={score_col: "macro_trend_score_continuous"}
                    )
                    nnz_before = int(before_block.notna().sum().sum())

                    # Align to training_index with a one-bar lag to avoid look-ahead,
                    # mirroring the convention used for other regime specialists.
                    block = before_block.shift(1).fillna(method="ffill")
                    block = block.reindex(training_index, method="ffill")
                    nnz_after = int(block.notna().sum().sum())

                    blocks.append(block)
                    log_success(
                        " Added Macro Trend specialist block from 'hmm_macro_trend_training_data_15m': "
                        f"shape={block.shape}, non_null_before={nnz_before}, non_null_after={nnz_after}"
                    )
                    if nnz_after == 0:
                        log_warning(
                            " Macro Trend block aligned to training_index is all-NaN. "
                            "Check macro alpha timestamps and values."
                        )
        else:
            log_info(" Skipping Macro Trend specialist outputs (enable_macro_trend_specialist=False)")
    except Exception as e:
        log_warning(f" Failed to load Macro Trend specialist outputs: {e}")

    # ------------------------------------------------------------------
    # 7) MR Trend specialist – XGB MR vs Trend classifier outputs
    # ------------------------------------------------------------------
    try:
        if config.get("enable_mr_trend_specialist", True):
            log_info("=" * 80)
            log_info(" LOADING SPECIALIST: MR TREND (XGB) OUTPUTS")
            log_info("=" * 80)

            mr_trend_context = {
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": base_timeframe,
                "direction": direction,
                "model": "regime_mr_trend",
                "step_name": "xgb_mr_trend_step",
            }

            mr_trend_training = artifact_router.load(
                artifact_name="xgb_mr_trend_training_data",
                artifact_type="data",
                data_category="features",
                context=mr_trend_context,
            )

            if mr_trend_training is not None and not getattr(mr_trend_training, "empty", True):
                if not isinstance(mr_trend_training, pd.DataFrame):
                    mr_trend_training = pd.DataFrame(mr_trend_training)
                mr_trend_training = _standardize_index(mr_trend_training)

                if (
                    isinstance(mr_trend_training.index, pd.DatetimeIndex)
                    and len(mr_trend_training.index) > 0
                ):
                    log_info(
                        " MR Trend (xgb_mr_trend_training_data) index range: %s → %s (n=%d)"
                        % (
                            mr_trend_training.index.min(),
                            mr_trend_training.index.max(),
                            len(mr_trend_training.index),
                        )
                    )

                if "target" in mr_trend_training.columns:
                    # Use the discrete regime label as a compact scalar and
                    # expose a helper binary flag marking explicit MR (class 2).
                    before_block = pd.DataFrame(
                        index=mr_trend_training.index,
                        data={
                            "mr_trend_state": mr_trend_training["target"].astype(float),
                            "mr_trend_is_mr": (mr_trend_training["target"] == 2).astype(float),
                        },
                    )
                    nnz_before = int(before_block.notna().sum().sum())

                    # Align to training_index with a one-bar lag, mirroring other
                    # regime specialists to avoid look-ahead.
                    block = before_block.shift(1).fillna(method="ffill")
                    block = block.reindex(training_index, method="ffill")
                    nnz_after = int(block.notna().sum().sum())

                    blocks.append(block)
                    log_success(
                        " Added MR Trend specialist block from 'xgb_mr_trend_training_data': "
                        f"shape={block.shape}, non_null_before={nnz_before}, "
                        f"non_null_after={nnz_after}"
                    )
                    if nnz_after == 0:
                        log_warning(
                            " MR Trend block aligned to training_index is all-NaN. "
                            "Check MR Trend timestamps and values."
                        )
        else:
            log_info(" Skipping MR Trend specialist outputs (enable_mr_trend_specialist=False)")
    except Exception as e:
        log_warning(f" Failed to load MR Trend specialist outputs: {e}")

    # ------------------------------------------------------------------
    # 9) SR labeling XGB specialist – S/R-based meta signal
    # ------------------------------------------------------------------
    try:
        if config.get("enable_sr_labeling_specialist", True):
            log_info("=" * 80)
            log_info(" LOADING SPECIALIST: SR LABELING XGB OUTPUTS")
            log_info("=" * 80)

            sr_context = {
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": base_timeframe,
                "direction": direction,
                "model": "sr_labeling_xgb",
                "step_name": "sr_labeling_xgb",
            }

            sr_artifact_name = f"sr_labeling_xgb_predictions_{base_timeframe}"
            log_info(f"Attempting to load SR labeling predictions: {sr_artifact_name}")
            sr_preds = artifact_router.load(
                artifact_name=sr_artifact_name,
                artifact_type="data",
                data_category="predictions",
                context=sr_context,
            )

            if sr_preds is not None and not getattr(sr_preds, "empty", True):
                if not isinstance(sr_preds, pd.DataFrame):
                    sr_preds = pd.DataFrame(sr_preds)
                sr_preds = _standardize_index(sr_preds)

                if isinstance(sr_preds.index, pd.DatetimeIndex) and len(sr_preds.index) > 0:
                    log_info(
                        " SR labeling (sr_labeling_xgb_predictions) index range: %s → %s (n=%d)"
                        % (
                            sr_preds.index.min(),
                            sr_preds.index.max(),
                            len(sr_preds.index),
                        )
                    )

                # Prefer an explicit sr_labeling_xgb_prob column; fall back to
                # generic probability-style names when necessary.
                prob_col = None
                for c in ("sr_labeling_xgb_prob", "predicted_prob", "predicted"):
                    if c in sr_preds.columns:
                        prob_col = c
                        break

                if prob_col is not None:
                    before_block = sr_preds[[prob_col]].copy()
                    before_block = before_block.rename(columns={prob_col: "sr_labeling_xgb_prob"})
                    nnz_before = int(before_block.notna().sum().sum())

                    # Align to training_index with a one-bar lag to avoid
                    # look-ahead: use the SR meta-signal available at t-1.
                    block = before_block.shift(1).fillna(method="ffill")
                    block = block.reindex(training_index, method="ffill")
                    nnz_after = int(block.notna().sum().sum())

                    blocks.append(block)
                    log_success(
                        " Added SR labeling specialist block from '%s': shape=%s, non_null_before=%d, non_null_after=%d"
                        % (sr_artifact_name, block.shape, nnz_before, nnz_after)
                    )
                    if nnz_after == 0:
                        log_warning(
                            " SR labeling block aligned to training_index is all-NaN. "
                            "Check sr_labeling_xgb prediction timestamps and values."
                        )
                else:
                    log_warning(
                        "⚠️ SR labeling predictions loaded but no probability-like column "
                        "found (expected one of: sr_labeling_xgb_prob, predicted_prob, predicted)."
                    )
        else:
            log_info(" Skipping SR labeling specialist outputs (enable_sr_labeling_specialist=False)")
    except Exception as e:
        log_warning(f" Failed to load SR labeling specialist outputs: {e}")

    # 8) Volume Force specialist – scalar directional prediction
    # ------------------------------------------------------------------
    try:
        log_info("=" * 80)
        log_info(" LOADING SPECIALIST: VOLUME FORCE OUTPUTS")
        log_info("=" * 80)

        vol_force_context = {
            "symbol": symbol,
            "exchange": exchange,
            "timeframe": base_timeframe,
            "direction": direction,
            "model": "volume_force",
            "step_name": "ml_volume_force_step",
        }

        vol_force_preds = artifact_router.load(
            artifact_name="ml_volume_force_predictions",
            artifact_type="data",
            data_category="predictions",
            context=vol_force_context,
        )

        if vol_force_preds is not None and not getattr(vol_force_preds, "empty", True):
            if not isinstance(vol_force_preds, pd.DataFrame):
                vol_force_preds = pd.DataFrame(vol_force_preds)
            vol_force_preds = _standardize_index(vol_force_preds)

            if isinstance(vol_force_preds.index, pd.DatetimeIndex) and len(vol_force_preds.index) > 0:
                log_info(
                    " Volume Force (ml_volume_force_predictions) index range: %s → %s (n=%d)"
                    % (
                        vol_force_preds.index.min(),
                        vol_force_preds.index.max(),
                        len(vol_force_preds.index),
                    )
                )

            # Expose 'predicted' (scalar 0-1) and rename to vol_force_scalar
            vol_cols: List[str] = []
            rename_map = {}

            # Legacy scalar support
            if "predicted" in vol_force_preds.columns:
                vol_cols.append("predicted")
                rename_map["predicted"] = "vol_force_scalar"
            elif "scalar_pred" in vol_force_preds.columns:
                vol_cols.append("scalar_pred")
                rename_map["scalar_pred"] = "vol_force_scalar"

            # New Multi-Target Support
            for target in ["breakout", "volatility", "trend"]:
                col = f"vol_force_{target}"
                if col in vol_force_preds.columns:
                    vol_cols.append(col)

            if vol_cols:
                before_block = vol_force_preds[vol_cols].copy()
                if rename_map:
                    before_block = before_block.rename(columns=rename_map)

                nnz_before = int(before_block.notna().sum().sum())
                # Align to training_index with forward-fill. Note: Volume Force predictions
                # are typically OOF or next-bar forecasts. If used as a feature for
                # ensemble models predicting the *same* horizon, we might use them directly.
                # However, consistent with other regime signals which are states *entering*
                # the bar, we should check if shift(1) is needed.
                # Standard practice here: if the artifact is indexed by prediction time (entry),
                # we don't shift. If indexed by result time, we shift.
                # MLVolumeForceStep saves predictions indexed by entry time.
                # To be safe and prevent lookahead if the ensemble is trained on t+1 target,
                # we usually use features available at t.
                # Most specialist blocks here use shift(1) + ffill.
                # We will follow the pattern: shift(1) to represent "value available at open/close of previous bar".
                block = before_block.shift(1).fillna(method="ffill")
                block = block.reindex(training_index, method="ffill")
                nnz_after = int(block.notna().sum().sum())

                blocks.append(block)
                log_success(
                    " Added Volume Force specialist block from 'ml_volume_force_predictions': "
                    f"shape={block.shape}, non_null_before={nnz_before}, "
                    f"non_null_after={nnz_after}"
                )
                if nnz_after == 0:
                    log_warning(
                        " Volume Force block aligned to training_index is all-NaN. "
                        "Check volume force timestamps."
                    )
    except Exception as e:
        log_warning(f" Failed to load Volume Force specialist outputs: {e}")

    # SMC specialist block: scalar regime predictions from MLSMCRegimeStep
    try:
        log_info("=" * 80)
        log_info(" LOADING SPECIALIST: SMC REGIME OUTPUTS")
        log_info("=" * 80)

        smc_context = {
            "symbol": symbol,
            "exchange": exchange,
            "timeframe": regime_timeframe,
            "direction": direction,
            "model": "smc_regime",
            "step_name": "ml_smc_regime_step",
        }

        smc = artifact_router.load(
            artifact_name="smc_predictions_with_confidence",
            artifact_type="data",
            data_category="predictions",
            context=smc_context,
        )

        if smc is not None and not getattr(smc, "empty", True):
            if not isinstance(smc, pd.DataFrame):
                smc = pd.DataFrame(smc)
            smc = _standardize_index(smc)

            if isinstance(smc.index, pd.DatetimeIndex) and len(smc.index) > 0:
                log_info(
                    " SMC (smc_predictions_with_confidence) index range: %s → %s (n=%d)"
                    % (
                        smc.index.min(),
                        smc.index.max(),
                        len(smc.index),
                    )
                )

            smc_cols: List[str] = []
            if "predicted" in smc.columns:
                smc_cols.append("predicted")
            extra_smc = [
                c
                for c in smc.columns
                if c != "predicted" and (c.startswith("prob_") or c.startswith("confidence_"))
            ]
            smc_cols.extend(extra_smc)

            if smc_cols:
                before_block = smc[smc_cols].copy()
                nnz_before = int(before_block.notna().sum().sum())
                block = before_block.reindex(training_index, method="ffill")
                block = block.add_prefix("smc_")
                nnz_after = int(block.notna().sum().sum())
                # Always keep the SMC block, even if currently all-NaN, so that
                # downstream consumers can see the feature structure and we can
                # debug alignment issues based on warnings instead of silently
                # discarding this specialist source.
                blocks.append(block)
                if nnz_after == 0:
                    log_warning(
                        " SMC block aligned to training_index is all-NaN. "
                        "Keeping SMC specialist features, but they will be effectively missing; "
                        "check SMC timestamps, index alignment, and values."
                    )
                else:
                    log_success(
                        " Added SMC specialist block from 'smc_predictions_with_confidence': "
                        f"shape={block.shape}, non_null_before={nnz_before}, "
                        f"non_null_after={nnz_after}"
                    )
    except Exception as e:
        log_warning(f" Failed to load SMC specialist outputs: {e}")

        if regime_timeframe != "1h":
            try:
                smc_context = {
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": "1h",
                    "direction": direction,
                    "model": "smc_regime",
                    "step_name": "ml_smc_regime_step",
                }

                smc = artifact_router.load(
                    artifact_name="smc_predictions_with_confidence",
                    artifact_type="data",
                    data_category="predictions",
                    context=smc_context,
                )

                if smc is not None and not getattr(smc, "empty", True):
                    if not isinstance(smc, pd.DataFrame):
                        smc = pd.DataFrame(smc)
                    smc = _standardize_index(smc)

                    smc_cols: List[str] = []
                    if "predicted" in smc.columns:
                        smc_cols.append("predicted")
                    extra_smc = [
                        c
                        for c in smc.columns
                        if c != "predicted" and (c.startswith("prob_") or c.startswith("confidence_"))
                    ]
                    smc_cols.extend(extra_smc)

                    if smc_cols:
                        before_block = smc[smc_cols].copy()
                        nnz_before = int(before_block.notna().sum().sum())
                        block = before_block.reindex(training_index, method="ffill")
                        block = block.add_prefix("smc_")
                        nnz_after = int(block.notna().sum().sum())
                        blocks.append(block)
                        if nnz_after == 0:
                            log_warning(
                                " SMC block aligned to training_index is all-NaN. "
                                "Keeping SMC specialist features, but they will be effectively missing; "
                                "check SMC timestamps, index alignment, and values."
                            )
                        else:
                            log_success(
                                " Added SMC specialist block from 'smc_predictions_with_confidence' (1h fallback): "
                                f"shape={block.shape}, non_null_before={nnz_before}, non_null_after={nnz_after}"
                            )
            except Exception as smc_fallback_exc:
                log_warning(f" Failed to load SMC specialist outputs (1h fallback): {smc_fallback_exc}")

    # Mean-reversion specialist block: mr_* features from MLMeanReversionRegimeStep
    try:
        if config.get("enable_mean_reversion_specialist", True):
            log_info("=" * 80)
            log_info(" LOADING SPECIALIST: MEAN-REVERSION OUTPUTS")
            log_info("=" * 80)

            mr_artifact_name = f"ml_mean_reversion_training_data_{regime_timeframe}"
            direction_norm = str(direction).lower()
            directions_to_try = [direction]
            if direction_norm not in {"long", "short"}:
                directions_to_try = ["long", "short"]

            mr = None
            mr_loaded_direction = None
            for dir_try in directions_to_try:
                mr_context = {
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": regime_timeframe,
                    "direction": dir_try,
                    "model": "mean_reversion",
                    "step_name": "ml_mean_reversion_step",
                }

                log_info(f"Attempting to load Mean Reversion data: {mr_artifact_name} (direction={dir_try})")
                try:
                    mr = artifact_router.load(
                        artifact_name=mr_artifact_name,
                        artifact_type="data",
                        data_category="features",
                        context=mr_context,
                    )
                    mr_loaded_direction = dir_try
                    if mr is not None and not getattr(mr, "empty", True):
                        break
                except FileNotFoundError:
                    mr = None
                    mr_loaded_direction = None
                    continue

            if mr is not None and not getattr(mr, "empty", True):
                if not isinstance(mr, pd.DataFrame):
                    mr = pd.DataFrame(mr)
                mr = _standardize_index(mr)

                if mr_loaded_direction is not None and str(mr_loaded_direction).lower() != direction_norm:
                    log_warning(
                        " Loaded mean-reversion artifact using fallback direction=%s (requested direction=%s)"
                        % (mr_loaded_direction, direction)
                    )

                if isinstance(mr.index, pd.DatetimeIndex) and len(mr.index) > 0:
                    log_info(
                        " Mean-reversion (%s) index range: %s → %s (n=%d)"
                        % (
                            mr_artifact_name,
                            mr.index.min(),
                            mr.index.max(),
                            len(mr.index),
                        )
                    )

                mr_cols = [c for c in mr.columns if c.startswith("mr_")]

                if mr_cols:
                    before_block = mr[mr_cols].copy()
                    nnz_before = int(before_block.notna().sum().sum())
                    block = before_block.reindex(training_index, method="ffill")
                    nnz_after = int(block.notna().sum().sum())
                    # Always keep the mean-reversion block, even if currently all-NaN,
                    # mirroring the SMC behavior so we don't silently lose this
                    # specialist source when alignment is misconfigured.
                    if nnz_after == 0:
                        log_warning(
                            " Mean-reversion block aligned to training_index is all-NaN. "
                            "Keeping mean-reversion specialist features, but they will be effectively missing; "
                            "check mean-reversion timestamps, index alignment, and values."
                        )
                    else:
                        blocks.append(block)
                        log_success(
                            f" Added Mean-reversion specialist block from '{mr_artifact_name}': "
                            f"shape={block.shape}, non_null_before={nnz_before}, "
                            f"non_null_after={nnz_after}"
                        )
        else:
            log_info(" Skipping Mean-reversion specialist outputs (enable_mean_reversion_specialist=False)")
    except Exception as e:
        log_warning(f" Failed to load Mean-reversion specialist outputs: {e}")

    if not blocks:
        msg = (
            " No specialist model outputs found (ML Risk / Liquidity / Breakout/Bounce). "
            "These features are strongly recommended for downstream models."
        )
        log_warning(msg)
        if strict:
            raise ValueError(msg)
        return None

    # Concatenate all specialist blocks along columns
    combined = pd.concat(blocks, axis=1)
    # Ensure final index exactly matches training_index
    combined = combined.reindex(training_index, method="ffill")

    # Optional projection: collapse specialist outputs to canonical scalar
    # features per specialist when requested by the caller. This is used by
    # training pipelines that want a single, interpretable scalar per
    # specialist rather than the full multi-column regime blocks.
    if config.get("use_canonical_specialist_scalars", False):
        combined = _project_specialists_to_canonical_scalars(combined)

    # Global coverage check: if the combined specialist frame is effectively
    # empty (almost all NaN after alignment), fail fast in strict mode so the
    # caller can fix timestamp/index alignment instead of silently training
    # without regime features.
    total_cells = combined.shape[0] * combined.shape[1]
    if total_cells > 0:
        nnz_combined = int(combined.notna().sum().sum())
        coverage = nnz_combined / total_cells
        if coverage < 0.01:
            msg = (
                "⚠️ Specialist outputs coverage below 1% after alignment: "
                f"non_null={nnz_combined}, total_cells={total_cells}, "
                f"coverage={coverage:.6f}. Check specialist timestamps and "
                "training_index alignment."
            )
            log_warning(msg)
            if strict:
                raise ValueError(
                    "Specialist outputs have insufficient coverage (<1%) after "
                    "alignment; aborting training to avoid silently dropping "
                    "all regime features."
                )

    return combined
