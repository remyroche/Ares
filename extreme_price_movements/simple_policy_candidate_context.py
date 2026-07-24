"""Strict context join for simple-policy research candidates.

The simple-policy candidate parquet is deliberately narrow.  The admitted
execution ledger is the authoritative source for the causal corrected-EV and
raw-Bayesian sizing inputs.  This module joins the two static ledgers by trade
identity rather than relying on their (usually identical) parquet row order.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

JOIN_KEY_COLUMNS = ("timestamp", "symbol", "side_name")
CORRECTED_EV_COLUMNS = (
    "threshold_basis_corrected_expected_ev",
    "threshold_basis_corrected_expected_ev_rank",
)
RAW_BAYESIAN_CONTEXT_COLUMNS = (
    "expected_net_ev_after_1pct_mlp_direct",
    "meta_hit_probability_uncertainty_p1mp",
    "gmm_ood_score",
    "cluster_entropy_norm",
)
EXECUTION_CONTEXT_COLUMNS = CORRECTED_EV_COLUMNS + RAW_BAYESIAN_CONTEXT_COLUMNS

_SOURCE_PROVENANCE_COLUMNS = (
    "threshold_basis_reference_asof",
    "threshold_basis_selected",
    "threshold_basis_mapped_expected_ev_valid",
    "threshold_basis_invalid_mapped_expected_ev_sentinel",
)


@dataclass(frozen=True)
class CandidateContextJoinAudit:
    candidate_rows: int
    source_rows: int
    matched_rows: int
    exact_source_coverage: bool
    source_positionally_aligned: bool
    earliest_timestamp_utc: str | None
    latest_timestamp_utc: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc(values: pd.Series, *, label: str) -> pd.Series:
    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"{label} contains {int(parsed.isna().sum())} invalid timestamps")
    return parsed


def _normalise_side(values: pd.Series, *, label: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    out = pd.Series(index=values.index, dtype="object")
    numeric_mask = numeric.notna()
    valid_numeric = numeric_mask & numeric.isin((-1, 1))
    out.loc[valid_numeric & numeric.eq(1)] = "long"
    out.loc[valid_numeric & numeric.eq(-1)] = "short"

    text = values.astype("string").str.strip().str.lower()
    text_mask = ~numeric_mask
    out.loc[text_mask & text.isin(("long", "buy"))] = "long"
    out.loc[text_mask & text.isin(("short", "sell"))] = "short"
    if out.isna().any():
        examples = values.loc[out.isna()].astype(str).drop_duplicates().head(3).tolist()
        raise ValueError(f"{label} contains invalid side values: {examples}")
    return out.astype("string")


def _canonical_keys(rows: pd.DataFrame, *, source: bool) -> pd.DataFrame:
    timestamp_candidates = ("__ts__", "timestamp", "signal_bar_ts") if source else (
        "timestamp",
        "signal_bar_ts",
        "__ts__",
    )
    symbol_candidates = ("__symbol__", "symbol") if source else ("symbol", "__symbol__")
    side_candidates = ("side_name", "side") if source else ("side", "side_name")

    def first(names: tuple[str, ...], kind: str) -> str:
        try:
            return next(name for name in names if name in rows.columns)
        except StopIteration as exc:
            raise ValueError(f"missing {kind} join column; expected one of {names}") from exc

    ts_col = first(timestamp_candidates, "timestamp")
    symbol_col = first(symbol_candidates, "symbol")
    side_col = first(side_candidates, "side")
    keys = pd.DataFrame(index=rows.index)
    keys["timestamp"] = _utc(rows[ts_col], label=ts_col)
    keys["symbol"] = rows[symbol_col].astype("string").str.strip()
    if keys["symbol"].isna().any() or keys["symbol"].eq("").any():
        raise ValueError(f"{symbol_col} contains missing or empty symbols")
    keys["side_name"] = _normalise_side(rows[side_col], label=side_col)

    # When aliases coexist, require them to identify the same static row.
    for alias in set(timestamp_candidates).intersection(rows.columns) - {ts_col}:
        if not keys["timestamp"].equals(_utc(rows[alias], label=alias)):
            raise ValueError(f"conflicting timestamp aliases: {ts_col} and {alias}")
    for alias in set(symbol_candidates).intersection(rows.columns) - {symbol_col}:
        other = rows[alias].astype("string").str.strip()
        if not keys["symbol"].equals(other):
            raise ValueError(f"conflicting symbol aliases: {symbol_col} and {alias}")
    for alias in set(side_candidates).intersection(rows.columns) - {side_col}:
        other = _normalise_side(rows[alias], label=alias)
        if not keys["side_name"].equals(other):
            raise ValueError(f"conflicting side aliases: {side_col} and {alias}")
    return keys


def _assert_unique(keys: pd.DataFrame, *, label: str) -> None:
    duplicates = keys.duplicated(list(JOIN_KEY_COLUMNS), keep=False)
    if duplicates.any():
        examples = keys.loc[duplicates, list(JOIN_KEY_COLUMNS)].head(3).to_dict("records")
        raise ValueError(
            f"{label} has {int(duplicates.sum())} rows with duplicate trade keys; examples={examples}"
        )


def _same_numeric(left: pd.Series, right: pd.Series) -> bool:
    a = pd.to_numeric(left, errors="coerce").to_numpy(dtype=np.float64)
    b = pd.to_numeric(right, errors="coerce").to_numpy(dtype=np.float64)
    return bool(np.allclose(a, b, rtol=1e-10, atol=1e-12, equal_nan=True))


def join_candidate_execution_context(
    candidates: pd.DataFrame,
    admitted_execution_ledger: pd.DataFrame,
    *,
    require_exact_source_coverage: bool = True,
) -> tuple[pd.DataFrame, CandidateContextJoinAudit]:
    """Attach authoritative EV/Bayesian context with a strict one-to-one join.

    The result retains the candidate index and row order.  Duplicates are
    rejected rather than silently collapsed because the key identifies one
    executable direction for one symbol/bar.  ``threshold_basis_reference_asof``
    is checked row-by-row to prevent a future admission state from being joined.
    """
    missing = [column for column in EXECUTION_CONTEXT_COLUMNS if column not in admitted_execution_ledger]
    if missing:
        raise ValueError(f"admitted execution ledger is missing context columns: {missing}")

    candidate_keys = _canonical_keys(candidates, source=False).reset_index(drop=True)
    source_keys = _canonical_keys(admitted_execution_ledger, source=True).reset_index(drop=True)
    _assert_unique(candidate_keys, label="candidate ledger")
    _assert_unique(source_keys, label="admitted execution ledger")

    source = source_keys.copy()
    for column in EXECUTION_CONTEXT_COLUMNS + _SOURCE_PROVENANCE_COLUMNS:
        if column in admitted_execution_ledger:
            source[column] = admitted_execution_ledger[column].reset_index(drop=True)

    candidate_identity = candidate_keys.copy()
    candidate_identity["__candidate_row__"] = np.arange(len(candidates), dtype=np.int64)
    joined = candidate_identity.merge(
        source,
        on=list(JOIN_KEY_COLUMNS),
        how="left",
        sort=False,
        validate="one_to_one",
        indicator=True,
    ).sort_values("__candidate_row__", kind="stable")
    unmatched = joined["_merge"].ne("both")
    if unmatched.any():
        examples = joined.loc[unmatched, list(JOIN_KEY_COLUMNS)].head(3).to_dict("records")
        raise ValueError(f"{int(unmatched.sum())} candidate rows lack admitted context; examples={examples}")

    candidate_index = pd.MultiIndex.from_frame(candidate_keys)
    source_index = pd.MultiIndex.from_frame(source_keys)
    exact_coverage = len(candidate_index) == len(source_index) and not source_index.difference(
        candidate_index
    ).size
    if require_exact_source_coverage and not exact_coverage:
        extra = source_index.difference(candidate_index)
        raise ValueError(f"admitted execution ledger has {len(extra)} extra trade keys")

    if "threshold_basis_reference_asof" in joined:
        reference_asof = _utc(
            joined["threshold_basis_reference_asof"], label="threshold_basis_reference_asof"
        )
        future = reference_asof.gt(joined["timestamp"])
        if future.any():
            raise ValueError(f"{int(future.sum())} rows use a future corrected-EV reference")
    if "threshold_basis_selected" in joined and not joined["threshold_basis_selected"].eq(True).all():
        raise ValueError("admitted execution ledger contains non-selected rows")
    if "threshold_basis_mapped_expected_ev_valid" in joined and not joined[
        "threshold_basis_mapped_expected_ev_valid"
    ].eq(True).all():
        raise ValueError("admitted execution ledger contains invalid mapped EV rows")
    if "threshold_basis_invalid_mapped_expected_ev_sentinel" in joined and joined[
        "threshold_basis_invalid_mapped_expected_ev_sentinel"
    ].eq(True).any():
        raise ValueError("admitted execution ledger contains invalid mapped-EV sentinels")

    out = candidates.copy()
    for column in EXECUTION_CONTEXT_COLUMNS:
        values = joined[column].reset_index(drop=True)
        if not np.isfinite(pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)).all():
            raise ValueError(f"joined context column {column!r} contains non-finite values")
        if column in out and not _same_numeric(out[column].reset_index(drop=True), values):
            raise ValueError(f"candidate column {column!r} conflicts with admitted execution ledger")
        out[column] = values.to_numpy()
    out["ev_rank_pct"] = pd.to_numeric(
        out["threshold_basis_corrected_expected_ev_rank"], errors="raise"
    ).clip(0.0, 1.0)

    audit = CandidateContextJoinAudit(
        candidate_rows=len(candidates),
        source_rows=len(admitted_execution_ledger),
        matched_rows=len(joined),
        exact_source_coverage=exact_coverage,
        source_positionally_aligned=candidate_index.equals(source_index),
        earliest_timestamp_utc=(str(candidate_keys["timestamp"].min()) if len(candidate_keys) else None),
        latest_timestamp_utc=(str(candidate_keys["timestamp"].max()) if len(candidate_keys) else None),
    )
    return out, audit


def load_candidate_execution_context(
    candidate_path: str | Path,
    admitted_execution_ledger_path: str | Path,
    *,
    require_exact_source_coverage: bool = True,
) -> tuple[pd.DataFrame, CandidateContextJoinAudit]:
    """Load parquet ledgers and apply :func:`join_candidate_execution_context`."""
    return join_candidate_execution_context(
        pd.read_parquet(candidate_path),
        pd.read_parquet(admitted_execution_ledger_path),
        require_exact_source_coverage=require_exact_source_coverage,
    )


__all__ = [
    "CORRECTED_EV_COLUMNS",
    "EXECUTION_CONTEXT_COLUMNS",
    "JOIN_KEY_COLUMNS",
    "RAW_BAYESIAN_CONTEXT_COLUMNS",
    "CandidateContextJoinAudit",
    "join_candidate_execution_context",
    "load_candidate_execution_context",
]
