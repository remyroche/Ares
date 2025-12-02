"""Live Regime Outputs Aggregator

This module provides a unified interface for loading and aligning the
predictions of specialist regime models for *live* trading usage.

It is designed to be called from trading components (signal pipeline,
regime monitor, risk manager) that already maintain a base timeframe
index (e.g. 15m bars). Given that index, it will:

- Load specialist regime outputs from versioned artifacts via
  ``ArtifactRouter``
- Align all available regime features to the provided index
- Return a single DataFrame with columns for each regime model

Covered specialist steps:

- ``hmm_ml_alpha_step``          → HMM Alpha scores
- ``ml_liquidity_regime_step``   → Liquidity regime probabilities (15m)
- ``ml_smc_regime_step``         → SMC breakout/breakdown scalar signal
- ``ml_breakout_bounce_regime_step`` → Breakout/Bounce probabilities + edge scores
- ``ml_risk_regime_step``        → ML risk regime probabilities
- ``ml_path_regime_step``        → Path-regime features (when available)
- ``ml_risk_regime_step`` (HMM flavor) → HMM-based risk regimes
- ``ml_mean_reversion_step``     → Mean-reversion probability / score

Implementation notes
--------------------

- For consistency with the training pipeline, this module reuses
  ``get_specialist_models_outputs`` for ML Risk, HMM Alpha, Liquidity,
  Breakout/Bounce, Path regimes, and ML Risk HMM regimes.
- It extends that logic with two additional live blocks:
  - SMC predictions from ``smc_predictions_with_confidence``
  - Mean-reversion outputs from ``ml_mean_reversion_training_data_*``
- All outputs are aligned to ``target_index`` via forward-fill.
  Callers that require strict no-lookahead behaviour can apply
  ``.shift(1)`` on the returned DataFrame.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from src.utils.artifact_router import ArtifactRouter
from src.utils.tprint import tprint_info, tprint_warning
from src.utils.ml_common.get_specialist_models_outputs import (
    get_specialist_models_outputs,
    _standardize_index,
)


def _load_smc_block(
    *,
    router: ArtifactRouter,
    symbol: str,
    exchange: str,
    direction: str,
    regime_timeframe: str,
    target_index: pd.DatetimeIndex,
    strict: bool,
) -> Optional[pd.DataFrame]:
    """Load SMC predictions and align to target_index.

    Source artifact: ``smc_predictions_with_confidence`` produced by
    ``MLSMCRegimeStep``.
    """

    context = {
        "symbol": symbol,
        "exchange": exchange,
        "timeframe": regime_timeframe,
        "direction": direction,
        "model": "smc_regime",
        "step_name": "ml_smc_regime_step",
    }

    try:
        smc = router.load(
            artifact_name="smc_predictions_with_confidence",
            artifact_type="data",
            data_category="predictions",
            context=context,
        )
    except FileNotFoundError:
        msg = (
            "SMC predictions artifact 'smc_predictions_with_confidence' not found "
            f"for {symbol}/{exchange} [{regime_timeframe}]"
        )
        if strict:
            raise
        tprint_warning(msg)
        return None
    except Exception as exc:  # pragma: no cover - defensive
        msg = f"Failed to load SMC predictions: {exc}"
        if strict:
            raise RuntimeError(msg) from exc
        tprint_warning(msg)
        return None

    if smc is None or getattr(smc, "empty", True):
        if strict:
            raise ValueError("SMC predictions artifact is empty")
        tprint_warning("SMC predictions artifact is empty; skipping SMC block")
        return None

    if not isinstance(smc, pd.DataFrame):
        smc = pd.DataFrame(smc)

    smc = _standardize_index(smc)
    if not isinstance(smc.index, pd.DatetimeIndex) or smc.index.empty:
        if strict:
            raise ValueError("SMC predictions artifact has no valid DatetimeIndex")
        tprint_warning("SMC predictions artifact has no valid DatetimeIndex; skipping")
        return None

    # Minimal contract: expose a single scalar score per bar.
    # Prefer an explicit 'predicted' column when present.
    cols: list[str] = []
    if "predicted" in smc.columns:
        cols.append("predicted")
    # Backward compatibility: keep any other SMC-derived columns if they exist.
    extra = [
        c
        for c in smc.columns
        if c != "predicted" and (c.startswith("prob_") or c.startswith("confidence_"))
    ]
    cols.extend(extra)

    if not cols:
        if strict:
            raise ValueError("SMC predictions contain no usable columns")
        tprint_warning("SMC predictions contain no usable columns; skipping")
        return None

    block = smc[cols].copy()

    # Optional isotonic calibration: map the scalar SMC score to an expected
    # forward return (clipped during training) and derive a normalized size
    # proxy in [-1, 1].
    if "predicted" in block.columns:
        iso = None
        try:
            iso = router.load(
                artifact_name="smc_scalar_isotonic_calibrator",
                artifact_type="model",
                data_category="models",
                context=context,
            )
        except FileNotFoundError:
            iso = None
        except Exception as exc:  # pragma: no cover - defensive
            tprint_warning(f"Failed to load SMC isotonic calibrator: {exc}")
            iso = None

        if iso is not None:
            try:
                pred_vals = block["predicted"].astype(float).to_numpy()
                expected = iso.predict(pred_vals)

                # Use the clip value stored on the calibrator when available,
                # otherwise fall back to the training default.
                ret_clip = getattr(iso, "return_clip_", 0.10)
                try:
                    ret_clip = float(ret_clip)
                except Exception:
                    ret_clip = 0.10
                if ret_clip <= 0.0:
                    ret_clip = 0.10

                size = expected / ret_clip

                expected_series = pd.Series(expected, index=block.index)
                size_series = pd.Series(size, index=block.index).clip(-1.0, 1.0)

                block["expected_return"] = expected_series
                block["size"] = size_series
            except Exception as exc:  # pragma: no cover - defensive
                tprint_warning(f"Failed to apply SMC isotonic calibrator: {exc}")

    # Align to target index with forward-fill.
    aligned = block.reindex(target_index, method="ffill")

    # Prefix columns for clarity when concatenated.
    aligned = aligned.add_prefix("smc_")
    return aligned


def _load_mean_reversion_block(
    *,
    router: ArtifactRouter,
    symbol: str,
    exchange: str,
    direction: str,
    regime_timeframe: str,
    target_index: pd.DatetimeIndex,
    strict: bool,
) -> Optional[pd.DataFrame]:
    """Load mean-reversion regime outputs and align to target_index.

    Source artifact: ``ml_mean_reversion_training_data_{regime_timeframe}``
    produced by ``MLMeanReversionRegimeStep``.
    """

    artifact_name = f"ml_mean_reversion_training_data_{regime_timeframe}"
    context = {
        "symbol": symbol,
        "exchange": exchange,
        "timeframe": regime_timeframe,
        "direction": direction,
        "model": "mean_reversion",
        "step_name": "ml_mean_reversion_step",
    }

    try:
        mr = router.load(
            artifact_name=artifact_name,
            artifact_type="data",
            data_category="features",
            context=context,
        )
    except FileNotFoundError:
        msg = (
            f"Mean-reversion training data artifact '{artifact_name}' not found "
            f"for {symbol}/{exchange} [{regime_timeframe}]"
        )
        if strict:
            raise
        tprint_warning(msg)
        return None
    except Exception as exc:  # pragma: no cover - defensive
        msg = f"Failed to load mean-reversion training data: {exc}"
        if strict:
            raise RuntimeError(msg) from exc
        tprint_warning(msg)
        return None

    if mr is None or getattr(mr, "empty", True):
        if strict:
            raise ValueError("Mean-reversion training data artifact is empty")
        tprint_warning("Mean-reversion training data artifact is empty; skipping block")
        return None

    if not isinstance(mr, pd.DataFrame):
        mr = pd.DataFrame(mr)

    mr = _standardize_index(mr)
    if not isinstance(mr.index, pd.DatetimeIndex) or mr.index.empty:
        if strict:
            raise ValueError("Mean-reversion artifact has no valid DatetimeIndex")
        tprint_warning("Mean-reversion artifact has no valid DatetimeIndex; skipping")
        return None

    # Use all mr_* columns as candidate signals (probability, score, etc.).
    mr_cols = [c for c in mr.columns if c.startswith("mr_")]
    if not mr_cols:
        if strict:
            raise ValueError("Mean-reversion artifact contains no 'mr_' columns")
        tprint_warning("Mean-reversion artifact contains no 'mr_' columns; skipping")
        return None

    block = mr[mr_cols].copy()
    aligned = block.reindex(target_index, method="ffill")
    return aligned


def load_live_regime_outputs(
    *,
    symbol: str,
    exchange: str,
    direction: str,
    base_timeframe: str,
    regime_timeframe: Optional[str] = None,
    target_index: pd.DatetimeIndex,
    config_overrides: Optional[Dict[str, Any]] = None,
    artifact_router: Optional[ArtifactRouter] = None,
    strict: bool = True,
) -> Optional[pd.DataFrame]:
    """Load and align live regime outputs for all specialist models.

    This function is the main entry point for trading code. It returns a
    single DataFrame indexed by ``target_index`` with columns from:

    - ML Risk regimes (probs + labels)
    - HMM Alpha scores
    - Liquidity regimes (probabilities)
    - Breakout/Bounce regimes + edge scores
    - Path regimes (when available)
    - ML Risk HMM flavour
    - SMC regimes (scalar signal from ``ml_smc_regime_step``)
    - Mean-reversion probability/scores (``ml_mean_reversion_step``)

    Args:
        symbol: Trading symbol (e.g. "ETHUSDT").
        exchange: Exchange name (e.g. "binance").
        direction: Trading direction (e.g. "long").
        base_timeframe: Base trading timeframe (e.g. "15m"). Used as the
            "timeframe" field when loading most artifacts.
        regime_timeframe: Regime detection timeframe (defaults to
            ``base_timeframe``). Used for regime-specific artifacts
            (risk, path, SMC, mean-reversion).
        target_index: DatetimeIndex to which all regime features will be
            aligned (typically the live bar index).
        config_overrides: Optional extra config keys passed through to
            ``get_specialist_models_outputs``.
        artifact_router: Optional preconfigured ``ArtifactRouter``. If
            not provided, a new instance is created with default paths.
        strict: If True, raise on missing artifacts; if False, return
            ``None`` when no regime sources are available.

    Returns:
        Combined DataFrame with specialist regime features aligned to
        ``target_index``, or ``None`` if no sources were found and
        ``strict=False``.
    """

    if not isinstance(target_index, pd.DatetimeIndex):
        target_index = pd.to_datetime(target_index, errors="coerce")

    if target_index.empty:
        if strict:
            raise ValueError("target_index is empty; cannot load regime outputs")
        tprint_warning("target_index is empty; returning None from load_live_regime_outputs")
        return None

    router = artifact_router or ArtifactRouter()
    regime_tf = str(regime_timeframe or base_timeframe)

    cfg: Dict[str, Any] = {
        "symbol": symbol,
        "exchange": exchange,
        "direction": direction,
        "timeframe": base_timeframe,
        "regime_timeframe": regime_tf,
    }
    if config_overrides:
        cfg.update(config_overrides)

    blocks: list[pd.DataFrame] = []

    # 1) Core specialist blocks (ML Risk, HMM Alpha, Liquidity, Breakout/Bounce,
    #    Path regimes, ML Risk HMM) via the shared loader.
    try:
        core = get_specialist_models_outputs(
            artifact_router=router,
            training_index=target_index,
            config=cfg,
            logger=None,
            strict=False,
        )
        if core is not None and not core.empty:
            blocks.append(core)
    except Exception as exc:  # pragma: no cover - defensive
        msg = f"Failed to load core specialist regime outputs: {exc}"
        if strict:
            raise RuntimeError(msg) from exc
        tprint_warning(msg)

    # 2) SMC predictions (classification-based breakout/breakdown regime).
    smc_block = _load_smc_block(
        router=router,
        symbol=symbol,
        exchange=exchange,
        direction=direction,
        regime_timeframe=regime_tf,
        target_index=target_index,
        strict=False if blocks else strict,
    )
    if smc_block is not None and not smc_block.empty:
        blocks.append(smc_block)

    # 3) Mean-reversion regime outputs.
    mr_block = _load_mean_reversion_block(
        router=router,
        symbol=symbol,
        exchange=exchange,
        direction=direction,
        regime_timeframe=regime_tf,
        target_index=target_index,
        strict=False if blocks else strict,
    )
    if mr_block is not None and not mr_block.empty:
        blocks.append(mr_block)

    if not blocks:
        msg = "No live regime outputs available from any specialist source"
        if strict:
            raise ValueError(msg)
        tprint_warning(msg)
        return None

    combined = pd.concat(blocks, axis=1)
    combined = combined.reindex(target_index, method="ffill")

    tprint_info(
        f"Loaded live regime outputs for {symbol}/{exchange} "
        f"[{base_timeframe}] with {combined.shape[1]} columns"
    )

    return combined
