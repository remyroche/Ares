"""Canonical OOF base candidate-population selection.

The population is ranked independently inside every UTC decision timestamp and
side.  It is intended to be materialized once and reused unchanged by the alpha
residual model, path auxiliary heads, and CatBoost path-archetype classifier.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .side_aware import candidate_id_series


@dataclass(frozen=True)
class BaseCandidatePopulationContract:
    top_fraction: float = 0.40
    timestamp_col: str = "__ts__"
    symbol_col: str = "__symbol__"
    side_col: str = "side_name"
    score_col: str = "score"
    timeframe: str = "1h"

    def validate(self) -> None:
        if not 0.0 < float(self.top_fraction) < 1.0:
            raise ValueError("top_fraction must be strictly between zero and one")
        if not str(self.timeframe).strip():
            raise ValueError("timeframe must be explicit and non-blank")


def _canonical_side(values: pd.Series) -> pd.Series:
    raw = values.astype(str).str.strip().str.lower()
    numeric = pd.to_numeric(values, errors="coerce")
    numeric_values = numeric.to_numpy(dtype=float, na_value=np.nan)
    side = pd.Series(
        np.where(numeric_values < 0.0, "short", "long"), index=values.index
    )
    side.loc[raw.isin(("short", "sell", "-1", "-1.0"))] = "short"
    side.loc[raw.isin(("long", "buy", "1", "1.0"))] = "long"
    invalid = values.isna() | ~(
        raw.isin(("short", "sell", "-1", "-1.0", "long", "buy", "1", "1.0"))
        | np.isfinite(numeric_values)
    )
    side.loc[invalid] = pd.NA
    return side.astype("string")


def select_base_candidate_population(
    frame: pd.DataFrame,
    contract: BaseCandidatePopulationContract = BaseCandidatePopulationContract(),
) -> pd.DataFrame:
    """Return exact top-fraction rows ranked within timestamp x side.

    Ties are deterministic: higher score first, then symbol lexical order.  The
    persisted integer rank is the selection authority; no downstream consumer
    is allowed to rerank this population.
    """

    contract.validate()
    required = {
        contract.timestamp_col,
        contract.symbol_col,
        contract.side_col,
        contract.score_col,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"base candidate source is missing columns: {missing}")
    work = frame.copy()
    work[contract.timestamp_col] = pd.to_datetime(
        work[contract.timestamp_col], utc=True, errors="coerce"
    )
    work[contract.side_col] = _canonical_side(work[contract.side_col])
    work[contract.score_col] = pd.to_numeric(work[contract.score_col], errors="coerce")
    valid = (
        work[contract.timestamp_col].notna()
        & work[contract.side_col].notna()
        & work[contract.symbol_col].notna()
        & np.isfinite(work[contract.score_col].to_numpy(dtype=float))
    )
    work = work.loc[valid].copy()
    work[contract.symbol_col] = work[contract.symbol_col].astype(str)
    work = work.sort_values(
        [contract.timestamp_col, contract.side_col, contract.score_col, contract.symbol_col],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    groups = work.groupby(
        [contract.timestamp_col, contract.side_col], sort=False, observed=True
    )
    work["base_candidate_rank_timestamp_side"] = (
        groups.cumcount().to_numpy(dtype=np.int32) + 1
    )
    work["base_candidate_group_rows"] = groups[contract.score_col].transform("size").astype(np.int32)
    keep_rows = np.ceil(
        work["base_candidate_group_rows"].to_numpy(dtype=float) * float(contract.top_fraction)
    ).astype(np.int32)
    selected = work["base_candidate_rank_timestamp_side"].to_numpy(dtype=np.int32) <= keep_rows
    work["base_candidate_rank_pct_timestamp_side"] = (
        work["base_candidate_rank_timestamp_side"].to_numpy(dtype=np.float64)
        / work["base_candidate_group_rows"].to_numpy(dtype=np.float64)
    ).astype(np.float32)
    selected_col = f"selected_top{int(round(100.0 * contract.top_fraction))}"
    work[selected_col] = selected
    work["candidate_handoff_rank_scope"] = "timestamp_side"
    return work.loc[selected].sort_values(
        [contract.timestamp_col, contract.symbol_col, contract.side_col], kind="mergesort"
    ).reset_index(drop=True)


def deterministic_candidate_ids(
    frame: pd.DataFrame,
    *,
    timestamp_col: str = "__ts__",
    symbol_col: str = "__symbol__",
    side_col: str = "side_name",
    timeframe: str = "1h",
) -> pd.Series:
    """Return stable IDs for the canonical UTC timestamp/symbol/side identity.

    This delegates to the shared side-aware candidate-ID contract used by path
    labels: ``symbol|UTC ISO Z|timeframe|canonical side``. It is deliberately
    independent of rank and source row position so OOF sources can prove they
    refer to the same candidate.
    """

    columns = (timestamp_col, symbol_col, side_col)
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"candidate identity columns are missing: {missing}")
    identity = frame.loc[:, list(columns)].copy()
    identity[timestamp_col] = pd.to_datetime(identity[timestamp_col], utc=True, errors="coerce")
    identity[symbol_col] = identity[symbol_col].astype("string").str.strip()
    identity[side_col] = _canonical_side(identity[side_col])
    invalid = (
        identity[timestamp_col].isna()
        | identity[symbol_col].isna()
        | identity[symbol_col].eq("")
        | identity[side_col].isna()
    )
    if invalid.any():
        raise ValueError("candidate identity contains null, blank, or invalid UTC values")
    if identity.duplicated(list(columns), keep=False).any():
        raise ValueError("candidate identity is not unique on timestamp/symbol/side")
    if not str(timeframe).strip():
        raise ValueError("candidate timeframe must be explicit and non-blank")
    result = candidate_id_series(
        identity[timestamp_col],
        identity[symbol_col],
        str(timeframe).strip(),
        identity[side_col],
    )
    result.index = frame.index
    return result.astype("string")


def candidate_identity_sha256(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str] = ("__ts__", "__symbol__", "side_name"),
) -> str:
    """Hash sorted UTC candidate identities for cross-stage equality audits."""

    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"candidate identity columns are missing: {missing}")
    identity = frame.loc[:, list(columns)].copy()
    identity[columns[0]] = pd.to_datetime(identity[columns[0]], utc=True, errors="raise")
    identity = identity.sort_values(list(columns), kind="mergesort")
    digest = hashlib.sha256()
    for row in identity.itertuples(index=False, name=None):
        digest.update(str(pd.Timestamp(row[0]).value).encode("ascii"))
        for value in row[1:]:
            digest.update(b"\x1f")
            digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()
