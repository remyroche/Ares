"""Canonical deterministic ranking for Stage-I economic tails.

The Stage-I pipeline evaluates one pooled global book after scores have been
mapped into common bps.  A score tie must therefore never inherit parquet row
order, fold order, or a caller's dataframe ordering.  This module centralises
the immutable identity order used by HPO, strict-OOF reports, admission, and
nested diagnostics.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd


RANKING_POLICY = "score_desc_side_candidate_id_asc_decision_ts_asc_v1"


def _utc_or_nat(values: Sequence[Any] | None, *, n: int) -> pd.Series:
    if values is None:
        return pd.Series(pd.NaT, index=np.arange(n), dtype="datetime64[ns, UTC]")
    result = pd.to_datetime(pd.Series(values), utc=True, errors="coerce")
    if len(result) != n:
        raise ValueError("ranking timestamps must be row-aligned")
    # Ranking identities must be total.  A finite candidate identity normally
    # resolves ties before time, but an absent time is still made deterministic.
    return result.fillna(pd.Timestamp.max.tz_localize("UTC"))


def _string_vector(
    values: Sequence[Any] | Any | None,
    *,
    n: int,
    label: str,
    default: str = "",
) -> np.ndarray:
    if values is None:
        return np.full(n, default, dtype=object)
    if isinstance(values, str):
        return np.full(n, values, dtype=object)
    result = pd.Series(values)
    if len(result) != n:
        raise ValueError(f"{label} must be row-aligned")
    if result.isna().any():
        raise ValueError(f"{label} must be non-null")
    return result.astype(str).to_numpy(dtype=object)


def stage_i_candidate_keys(
    candidate_ids: Sequence[Any], *, side_names: Sequence[Any] | Any | None = None
) -> np.ndarray:
    """Return immutable side-qualified candidate identities.

    Candidate IDs are the primary tie key.  Side qualification makes the key
    globally unique for pooled long/short reports even if an upstream source
    reuses an ID across sides.
    """
    ids = _string_vector(candidate_ids, n=len(candidate_ids), label="candidate_ids")
    sides = _string_vector(side_names, n=len(ids), label="side_names", default="")
    return np.asarray(
        [f"{side}::{candidate_id}" for side, candidate_id in zip(sides, ids)],
        dtype=object,
    )


def stable_stage_i_topk_positions(
    score: Sequence[float],
    *,
    candidate_ids: Sequence[Any],
    decision_timestamps: Sequence[Any] | None = None,
    side_names: Sequence[Any] | Any | None = None,
    signal_timestamps: Sequence[Any] | None = None,
    symbols: Sequence[Any] | None = None,
    count: int,
    valid_mask: Sequence[bool] | None = None,
) -> np.ndarray:
    """Select a score-descending top-k with immutable, input-order-free ties.

    The fallback fields are deliberately all decision-time identity fields;
    outcome columns are never considered.  ``candidate_ids`` must be unique
    within side, otherwise a complete deterministic identity does not exist.
    """
    values = np.asarray(score, dtype=np.float64).reshape(-1)
    n = len(values)
    ids = _string_vector(candidate_ids, n=n, label="candidate_ids")
    sides = _string_vector(side_names, n=n, label="side_names", default="")
    decision = _utc_or_nat(decision_timestamps, n=n)
    signal = _utc_or_nat(signal_timestamps, n=n)
    symbol = _string_vector(symbols, n=n, label="symbols", default="")
    if valid_mask is None:
        valid = np.isfinite(values)
    else:
        valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
        if len(valid) != n:
            raise ValueError("valid_mask must be row-aligned")
        valid &= np.isfinite(values)
    positions = np.flatnonzero(valid)
    if not len(positions):
        return np.asarray([], dtype=np.int64)
    if int(count) < 1:
        return np.asarray([], dtype=np.int64)
    keys = stage_i_candidate_keys(ids, side_names=sides)
    selected_keys = keys[positions]
    if len(pd.unique(selected_keys)) != len(selected_keys):
        raise ValueError("Stage-I ranking requires unique side-qualified candidate identities")
    work = pd.DataFrame(
        {
            "__position__": positions,
            "__score__": values[positions],
            "__candidate_key__": selected_keys,
            "__decision_ts__": decision.iloc[positions].to_numpy(),
            "__signal_ts__": signal.iloc[positions].to_numpy(),
            "__symbol__": symbol[positions],
        }
    )
    ordered = work.sort_values(
        ["__score__", "__candidate_key__", "__decision_ts__", "__signal_ts__", "__symbol__"],
        ascending=[False, True, True, True, True],
        kind="stable",
    )
    return ordered["__position__"].head(min(int(count), len(ordered))).to_numpy(dtype=np.int64)


def stable_stage_i_rank_frame(
    frame: pd.DataFrame,
    *,
    score_column: str,
    candidate_id_column: str = "candidate_id",
    side_column: str = "side_name",
    decision_column: str = "decision_ts",
    signal_column: str = "__ts__",
    symbol_column: str = "__symbol__",
) -> pd.DataFrame:
    """Return finite-score rows in the canonical Stage-I rank order."""
    if score_column not in frame or candidate_id_column not in frame:
        raise ValueError("Stage-I ranking requires score and candidate identity columns")
    score = pd.to_numeric(frame[score_column], errors="coerce").to_numpy(dtype=float)
    ids = frame[candidate_id_column].to_numpy(dtype=object)
    sides = frame[side_column].to_numpy(dtype=object) if side_column in frame else None
    decision = frame[decision_column] if decision_column in frame else None
    signal = frame[signal_column] if signal_column in frame else None
    symbols = frame[symbol_column].to_numpy(dtype=object) if symbol_column in frame else None
    positions = stable_stage_i_topk_positions(
        score,
        candidate_ids=ids,
        side_names=sides,
        decision_timestamps=decision,
        signal_timestamps=signal,
        symbols=symbols,
        count=int(np.isfinite(score).sum()),
    )
    return frame.iloc[positions].copy()


__all__ = [
    "RANKING_POLICY",
    "stage_i_candidate_keys",
    "stable_stage_i_rank_frame",
    "stable_stage_i_topk_positions",
]
