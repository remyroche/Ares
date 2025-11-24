"""Shared loader for specialist model outputs (ML Risk, HMM Alpha, Liquidity, Breakout/Bounce).

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

import numpy as np
import pandas as pd

from src.utils.tprint import tprint_info, tprint_warning, tprint_success


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

    # If we already had a DatetimeIndex, leave it as-is
    return df


def get_specialist_models_outputs(
    *,
    artifact_router: Any,
    training_index: pd.DatetimeIndex,
    config: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
    strict: bool = True,
) -> Optional[pd.DataFrame]:
    """Load specialist model outputs (ML Risk, HMM Alpha, Liquidity) and
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
    regime_timeframe = str(config.get("regime_timeframe") or base_timeframe)

    # Ensure training_index is a DatetimeIndex (naive) for reindexing.
    if not isinstance(training_index, pd.DatetimeIndex):
        training_index = pd.to_datetime(training_index, errors="coerce")

    # Basic diagnostics on the target index used for alignment
    if isinstance(training_index, pd.DatetimeIndex) and len(training_index) > 0:
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
                    if c.startswith("risk_regime") or c.startswith("risk_pred_")
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
    except Exception as e:
        log_warning(f"⚠️ Failed to load ML Risk specialist outputs: {e}")

    # ------------------------------------------------------------------
    # 2) HMM Alpha regimes (1h expectations → aligned to base timeframe)
    # ------------------------------------------------------------------
    try:
        log_info("=" * 80)
        log_info("🌟 LOADING SPECIALIST: HMM ALPHA OUTPUTS")
        log_info("=" * 80)

        alpha_context = {
            "symbol": symbol,
            "exchange": exchange,
            "timeframe": regime_timeframe,
            "direction": direction,
            "model": "regime_alpha",
            "step_name": "hmm_ml_alpha_step",
        }
        alpha_training = artifact_router.load(
            artifact_name="hmm_alpha_training_data_15m",
            artifact_type="data",
            data_category="features",
            context=alpha_context,
        )

        if alpha_training is not None and not getattr(alpha_training, "empty", True):
            if not isinstance(alpha_training, pd.DataFrame):
                alpha_training = pd.DataFrame(alpha_training)
            alpha_training = _standardize_index(alpha_training)

            if isinstance(alpha_training.index, pd.DatetimeIndex) and len(alpha_training.index) > 0:
                log_info(
                    "📈 HMM Alpha (hmm_alpha_training_data_15m) index range: %s → %s (n=%d)"
                    % (
                        alpha_training.index.min(),
                        alpha_training.index.max(),
                        len(alpha_training.index),
                    )
                )

            # Canonical downstream signal for HMM Alpha is the calibrated
            # continuous score produced by hmm_ml_alpha_step. When available,
            # also expose EWM-smoothed variants as additional specialist
            # channels so downstream models can choose between raw and
            # smoother alpha scores.
            alpha_cols: List[str] = []

            if "alpha_score_continuous" in alpha_training.columns:
                alpha_cols.append("alpha_score_continuous")

                ewm_cols = [
                    c
                    for c in alpha_training.columns
                    if c.startswith("alpha_score_continuous_ewm_")
                ]
                if ewm_cols:
                    # Stable ordering by period suffix for reproducibility
                    alpha_cols.extend(sorted(ewm_cols))
            else:
                # Backward compatibility: fall back to the primary alpha_pred_*
                # column if the unified score is not present (older artifacts).
                legacy_score_cols = [
                    c
                    for c in alpha_training.columns
                    if c.startswith("alpha_pred_")
                ]
                if legacy_score_cols:
                    alpha_cols.append(legacy_score_cols[0])

            if alpha_cols:
                before_block = alpha_training[alpha_cols].copy()
                nnz_before = int(before_block.notna().sum().sum())
                block = before_block.shift(1).fillna(method="ffill")
                block = block.reindex(training_index, method="ffill")
                nnz_after = int(block.notna().sum().sum())
                blocks.append(block)
                log_success(
                    "✅ Added HMM Alpha specialist block from 'hmm_alpha_training_data_15m': "
                    f"shape={block.shape}, non_null_before={nnz_before}, "
                    f"non_null_after={nnz_after}"
                )
                if nnz_after == 0:
                    log_warning(
                        "⚠️ HMM Alpha block aligned to training_index is all-NaN. "
                        "Check alpha_training timestamps and values."
                    )
    except Exception as e:
        log_warning(f"⚠️ Failed to load HMM Alpha specialist outputs: {e}")

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

        liquidity_probs = artifact_router.load(
            artifact_name="ml_liquidity_regime_probs_15m",
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
                block = before_block.reindex(training_index, method="ffill")
                nnz_after = int(block.notna().sum().sum())
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
        log_warning(f"⚠️ Failed to load Liquidity specialist outputs: {e}")

    # ------------------------------------------------------------------
    # 4) Breakout/Bounce regimes – 1h probabilities + edge scores
    # ------------------------------------------------------------------
    try:
        log_info("=" * 80)
        log_info("🔥 LOADING SPECIALIST: BREAKOUT/BOUNCE REGIME OUTPUTS")
        log_info("=" * 80)

        breakout_context = {
            "symbol": symbol,
            "exchange": exchange,
            "timeframe": regime_timeframe,
            "direction": direction,
            "model": "breakout_bounce",
            "step_name": "ml_breakout_bounce_regime_step",
        }

        breakout_training = artifact_router.load(
            artifact_name="ml_breakout_bounce_training_data_15m",
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
                    "📈 Breakout/Bounce (ml_breakout_bounce_training_data_15m) index range: %s → %s (n=%d)"
                    % (
                        breakout_training.index.min(),
                        breakout_training.index.max(),
                        len(breakout_training.index),
                    )
                )

            breakout_cols = [
                c
                for c in breakout_training.columns
                if c
                in [
                    "breakout_regime_0_prob",
                    "breakout_regime_1_prob",
                    "breakout_regime_2_prob",
                    "breakout_long_edge_score",
                    "breakout_short_edge_score",
                    "is_resistance",
                    "is_support",
                ]
            ]

            if breakout_cols:
                before_block = breakout_training[breakout_cols].copy()
                nnz_before = int(before_block.notna().sum().sum())
                block = before_block.shift(1).fillna(method="ffill")
                block = block.reindex(training_index, method="ffill")
                nnz_after = int(block.notna().sum().sum())
                blocks.append(block)
                log_success(
                    "✅ Added Breakout/Bounce specialist block from 'ml_breakout_bounce_training_data_15m': "
                    f"shape={block.shape}, non_null_before={nnz_before}, "
                    f"non_null_after={nnz_after}"
                )
                if nnz_after == 0:
                    log_warning(
                        "⚠️ Breakout/Bounce block aligned to training_index is all-NaN. "
                        "Check breakout_training timestamps and values."
                    )
    except Exception as e:
        log_warning(f"⚠️ Failed to load Breakout/Bounce specialist outputs: {e}")

    if not blocks:
        msg = (
            "❌ No specialist model outputs found (ML Risk / HMM Alpha / Liquidity / Breakout/Bounce). "
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

    return combined
