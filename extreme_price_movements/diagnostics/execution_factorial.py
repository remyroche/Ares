"""Paired diagnostics for execution-policy component substitutions.

The helpers in this module are intentionally policy-neutral.  They enforce a
fixed row population and summarize paired outcomes without fitting or tuning a
model or an execution parameter.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


EXIT_TYPES = (
    "timeout",
    "full_stop",
    "hard_tp",
    "trailing",
    "capital_exit",
    "adverse_exit",
)


def assert_fixed_stage_keys(
    ledger: pd.DataFrame,
    *,
    key_columns: Iterable[str],
    stage_column: str = "variant",
) -> None:
    """Require exactly the same unique row keys in every variant."""

    keys = list(key_columns)
    if ledger.duplicated([stage_column, *keys]).any():
        raise ValueError("Factorial ledger contains duplicate variant/row keys")
    expected: pd.MultiIndex | None = None
    for stage, part in ledger.groupby(stage_column, observed=True, sort=False):
        current = pd.MultiIndex.from_frame(part[keys]).sort_values()
        if expected is None:
            expected = current
        elif not current.equals(expected):
            raise ValueError(f"Variant {stage!r} does not use the fixed row population")


def mfe_capture_ratio(realized: np.ndarray, mfe: np.ndarray) -> np.ndarray:
    """Signed realized gross return divided by favorable excursion."""

    realized = np.asarray(realized, dtype=np.float64)
    mfe = np.asarray(mfe, dtype=np.float64)
    return np.divide(
        realized,
        mfe,
        out=np.full(realized.shape, np.nan, dtype=np.float64),
        where=np.isfinite(realized) & np.isfinite(mfe) & (mfe > 1e-12),
    )


def paired_variant_summary(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
) -> dict[str, Any]:
    """Summarize one candidate against an aligned reference frame."""

    if len(candidate) != len(reference):
        raise ValueError("Candidate/reference row counts differ")
    gross = pd.to_numeric(candidate["gross_return"], errors="coerce").to_numpy(float)
    net = pd.to_numeric(candidate["net_return"], errors="coerce").to_numpy(float)
    ref_gross = pd.to_numeric(reference["gross_return"], errors="coerce").to_numpy(float)
    delta = gross - ref_gross
    reason = candidate["exit_type"].astype(str).reset_index(drop=True)
    ref_reason = reference["exit_type"].astype(str).reset_index(drop=True)
    mfe = pd.to_numeric(candidate["mfe"], errors="coerce").to_numpy(float)
    capture = mfe_capture_ratio(gross, mfe)
    result: dict[str, Any] = {
        "rows": int(len(candidate)),
        "gross_ev": float(np.nanmean(gross)),
        "net_ev": float(np.nanmean(net)),
        "ev_delta": float(np.nanmean(delta)),
        "win_rate": float(np.nanmean(net > 0.0)),
        "win_rate_delta": float(np.nanmean(net > 0.0) - np.nanmean(
            pd.to_numeric(reference["net_return"], errors="coerce").to_numpy(float) > 0.0
        )),
        "mfe_capture_mean": float(np.nanmean(capture)),
        "mfe_capture_ratio_of_sums": float(np.nansum(gross) / max(np.nansum(mfe), 1e-12)),
        "mean_mfe": float(np.nanmean(mfe)),
        "mean_mae": float(pd.to_numeric(candidate["mae"], errors="coerce").mean()),
        "mean_holding_minutes": float(pd.to_numeric(candidate["holding_minutes"], errors="coerce").mean()),
        "holding_delta_minutes": float(
            pd.to_numeric(candidate["holding_minutes"], errors="coerce").mean()
            - pd.to_numeric(reference["holding_minutes"], errors="coerce").mean()
        ),
        "exit_type_changes": int(reason.ne(ref_reason).sum()),
        "return_change_gt_25bps": int(np.sum(np.abs(delta) > 0.0025)),
        "return_change_gt_50bps": int(np.sum(np.abs(delta) > 0.0050)),
        "return_change_gt_100bps": int(np.sum(np.abs(delta) > 0.0100)),
    }
    for exit_type in EXIT_TYPES:
        result[f"{exit_type}_rate"] = float(reason.eq(exit_type).mean())
        result[f"{exit_type}_delta"] = float(
            reason.eq(exit_type).mean() - ref_reason.eq(exit_type).mean()
        )
    return result


def exit_transition_matrix(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    variant: str,
) -> pd.DataFrame:
    """Return counts and reference-row-normalized exit transitions."""

    table = pd.crosstab(
        reference["exit_type"].astype(str).reset_index(drop=True),
        candidate["exit_type"].astype(str).reset_index(drop=True),
        dropna=False,
    ).reindex(index=EXIT_TYPES, columns=EXIT_TYPES, fill_value=0)
    rows: list[dict[str, Any]] = []
    for source in EXIT_TYPES:
        denominator = int(table.loc[source].sum())
        for target in EXIT_TYPES:
            count = int(table.loc[source, target])
            rows.append({
                "variant": variant,
                "reference_exit_type": source,
                "candidate_exit_type": target,
                "count": count,
                "reference_row_fraction": count / denominator if denominator else np.nan,
            })
    return pd.DataFrame(rows)


def interaction_delta(
    ev_pair: float,
    ev_a: float,
    ev_b: float,
    ev_reference: float,
) -> float:
    return float(ev_pair - ev_a - ev_b + ev_reference)
