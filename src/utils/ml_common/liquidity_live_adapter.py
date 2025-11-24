"""Live adapter for rule-based liquidity regimes.

This module provides a thin, production-oriented interface for consuming
liquidity regime probabilities in live or near-live contexts.

It is intentionally lightweight and reuses the existing artifact routing
and specialist loading patterns:

- Offline, `ml_liquidity_regime_step` produces a canonical
  `ml_liquidity_regime_probs_15m` artifact with per-regime probabilities.
- This adapter loads that artifact via `ArtifactRouter` and aligns the
  probabilities to an arbitrary target index (e.g. live 15m bars).

Usage example (inside a live trading component)::

    from datetime import datetime
    import pandas as pd
    from src.utils.artifact_router import ArtifactRouter
    from src.utils.ml_common.liquidity_live_adapter import (
        load_liquidity_regime_probs_for_index,
    )

    router = ArtifactRouter()
    # target_index should be the DatetimeIndex of your live 15m bars
    target_index = pd.date_range("2025-01-01", periods=100, freq="15T")

    probs = load_liquidity_regime_probs_for_index(
        symbol="ETHUSDT",
        exchange="binance",
        direction="long",
        base_timeframe="15m",
        target_index=target_index,
        artifact_router=router,
        strict=False,
    )

This keeps all regime construction logic in `MLLiquidityRegimeStep` and
exposes a clean, side-effect-free read API for live systems.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from src.utils.artifact_router import ArtifactRouter
from src.utils.tprint import tprint_info, tprint_warning
from src.utils.ml_common.get_specialist_models_outputs import _standardize_index


def load_liquidity_regime_probs_for_index(
    *,
    symbol: str,
    exchange: str,
    direction: str,
    base_timeframe: str,
    target_index: pd.DatetimeIndex,
    artifact_router: Optional[ArtifactRouter] = None,
    strict: bool = True,
) -> Optional[pd.DataFrame]:
    """Load 15m liquidity regime probabilities and align to a target index.

    This helper is designed for live or near-live usage: it reads the
    precomputed `ml_liquidity_regime_probs_15m` artifact produced by
    `MLLiquidityRegimeStep` and reindexes it to the provided
    `target_index` via forward-fill (optionally with a 1-bar shift to
    avoid lookahead in downstream consumers).

    Args:
        symbol: Trading symbol (e.g. "ETHUSDT").
        exchange: Exchange name (e.g. "binance").
        direction: Trading direction (e.g. "long" or "short").
        base_timeframe: Base timeframe of the live system (typically
            "15m" when consuming `ml_liquidity_regime_probs_15m`).
        target_index: DatetimeIndex to which regime probabilities should
            be aligned (e.g. your live 15m bar timestamps).
        artifact_router: Optional preconfigured ArtifactRouter. If not
            provided, a new router will be constructed with default
            paths.
        strict: If True, raise if artifacts are missing or empty. If
            False, return None on missing data.

    Returns:
        DataFrame indexed by `target_index` with columns matching the
        stored liquidity probabilities (e.g. `liquidity_regime_0_prob`,
        ..., `liquidity_regime_4_prob`), or None if strict=False and no
        data is available.
    """

    if not isinstance(target_index, pd.DatetimeIndex):
        target_index = pd.to_datetime(target_index, errors="coerce")

    if target_index.empty:
        if strict:
            raise ValueError("target_index is empty; cannot align liquidity regimes")
        return None

    router = artifact_router or ArtifactRouter()

    # Canonical artifact name for 15m probabilities produced by
    # ml_liquidity_regime_step. For non-15m base timeframes, the
    # probabilities are still native 15m and can be joined/aggregated by
    # the caller as needed.
    artifact_name = "ml_liquidity_regime_probs_15m"

    context = {
        "symbol": symbol,
        "exchange": exchange,
        "timeframe": base_timeframe,
        "direction": direction,
        "model": "liquidity_regime",
        "step_name": "ml_liquidity_regime_step",
    }

    tprint_info(
        f"Loading liquidity regime probabilities from '{artifact_name}' "
        f"for {symbol}/{exchange} [{base_timeframe}] {direction}/liquidity_regime"
    )

    try:
        probs = router.load(
            artifact_name=artifact_name,
            artifact_type="data",
            data_category="features",
            context=context,
        )
    except FileNotFoundError:
        msg = (
            f"Liquidity regime probabilities artifact '{artifact_name}' "
            f"not found for {symbol}/{exchange} [{base_timeframe}]"
        )
        if strict:
            raise FileNotFoundError(msg)
        tprint_warning(msg)
        return None
    except Exception as exc:
        msg = f"Failed to load liquidity regime probabilities: {exc}"
        if strict:
            raise RuntimeError(msg) from exc
        tprint_warning(msg)
        return None

    if probs is None or getattr(probs, "empty", True):
        msg = (
            f"Liquidity regime probabilities '{artifact_name}' are empty "
            f"for {symbol}/{exchange} [{base_timeframe}]"
        )
        if strict:
            raise ValueError(msg)
        tprint_warning(msg)
        return None

    if not isinstance(probs, pd.DataFrame):
        probs = pd.DataFrame(probs)

    # Standardize index using the same helper as specialist loaders.
    probs = _standardize_index(probs)

    if not isinstance(probs.index, pd.DatetimeIndex) or probs.index.empty:
        msg = "Liquidity probabilities artifact has no valid DatetimeIndex after standardization"
        if strict:
            raise ValueError(msg)
        tprint_warning(msg)
        return None

    # Select regime probability columns. These mirror the naming
    # convention used in ml_liquidity_regime_step.
    prob_cols = [
        c
        for c in probs.columns
        if c.startswith("liquidity_regime_") and c.endswith("_prob")
    ]
    if not prob_cols:
        msg = "No liquidity_regime_*_prob columns found in probabilities artifact"
        if strict:
            raise ValueError(msg)
        tprint_warning(msg)
        return None

    block = probs[prob_cols].copy()

    # Align to target_index via forward-fill so that each live bar sees
    # the most recent known liquidity regime probabilities. We do not
    # shift here; callers that require strict no-lookahead behaviour can
    # apply an additional `.shift(1)` on the returned frame.
    aligned = block.reindex(target_index, method="ffill")

    return aligned
