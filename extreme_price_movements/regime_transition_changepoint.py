"""Causal compact BOCPD context for transition-onset research.

This module is intentionally *not* a state-discovery system.  It turns a
small, predeclared set of observable market-transition fields into four online
break summaries which may be appended to the existing onset feature matrix.
The normalisation reference is an initial historical warm-up for each
continuous segment; no label, fitted state, or later observation can affect a
subsequent score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from extreme_price_movements.bayesian_changepoint import BOCPDConfig, bocpd_student_t


# These fields were selected before the ablation for their distinct observable
# mechanisms: breadth, dispersion/correlation, leverage/funding, covering and
# recovery.  They must not be selected against onset outcomes.
CHANGEPOINT_INPUT_COLUMNS: tuple[str, ...] = (
    "negative_breadth_pct",
    "breadth_dispersion",
    "correlation_breakdown_dispersion",
    "funding_deleveraging_divergence",
    "short_covering_score_market",
    "flush_recovery_state",
)
CHANGEPOINT_FEATURE_COLUMNS: tuple[str, ...] = (
    "bocpd_context__mean_probability",
    "bocpd_context__max_probability",
    "bocpd_context__break_count_ge_0_10",
    "bocpd_context__break_count_ge_0_25",
)


@dataclass(frozen=True)
class CausalChangePointConfig:
    """Fixed, bounded BOCPD configuration for the compact context block."""

    warmup_hours: int = 720
    expected_run_hours: int = 48
    max_run_hours: int = 96
    score_clip: float = 8.0


def _continuous_runs(frame: pd.DataFrame) -> list[np.ndarray]:
    """Return exact-hour runs without allowing a score to cross a gap."""

    required = {"source_utc", "segment_id"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"transition panel missing {sorted(missing)}")
    timestamp = pd.to_datetime(frame["source_utc"], utc=True)
    result: list[np.ndarray] = []
    for _, positions in frame.groupby("segment_id", observed=True, sort=False).groups.items():
        loc = np.asarray(list(positions), dtype=np.int64)
        loc = loc[np.argsort(timestamp.iloc[loc].to_numpy())]
        start = 0
        for end in range(1, len(loc) + 1):
            boundary = end == len(loc)
            if not boundary:
                boundary = timestamp.iloc[loc[end]] - timestamp.iloc[loc[end - 1]] != pd.Timedelta(hours=1)
            if boundary:
                result.append(loc[start:end])
                start = end
    return result


def _causal_scaled_scores(
    values: np.ndarray,
    *,
    config: CausalChangePointConfig,
) -> np.ndarray:
    """Score one continuous raw series with only its preceding warm-up fit.

    Scores in the warm-up window are deliberately missing: their robust scale
    would otherwise use future bars relative to early timestamps.  By the
    first non-missing score the entire warm-up is in the past.
    """

    output = np.full(len(values), np.nan, dtype=np.float32)
    warmup = int(config.warmup_hours)
    if len(values) <= warmup:
        return output
    raw = np.asarray(values, dtype=np.float64)
    reference = raw[:warmup]
    finite = reference[np.isfinite(reference)]
    if len(finite) < max(64, warmup // 8):
        return output
    median = float(np.median(finite))
    q25, q75 = np.quantile(finite, (0.25, 0.75))
    scale = max(float(q75 - q25), 1e-4)
    scaled = np.clip((np.nan_to_num(raw, nan=median) - median) / scale, -float(config.score_clip), float(config.score_clip))
    score = bocpd_student_t(
        scaled,
        BOCPDConfig(
            expected_run_hours=int(config.expected_run_hours),
            max_run_hours=int(config.max_run_hours),
        ),
    )
    output[warmup:] = score[warmup:]
    return output


def materialize_causal_changepoint_context(
    frame: pd.DataFrame,
    *,
    input_columns: Sequence[str] = CHANGEPOINT_INPUT_COLUMNS,
    config: CausalChangePointConfig = CausalChangePointConfig(),
) -> tuple[pd.DataFrame, list[str]]:
    """Materialize four fixed BOCPD summaries without outcome inputs.

    A run receives an independent online posterior, so a data gap or segment
    boundary cannot leak a previous market regime into the next one.  Fixed
    posterior levels (0.10 and 0.25) are design constants, rather than
    quantiles fitted using later or evaluation scores.
    """

    if int(config.warmup_hours) < 64:
        raise ValueError("warmup_hours must be at least 64")
    missing = [name for name in input_columns if name not in frame]
    if missing:
        raise KeyError(f"transition panel missing changepoint inputs {missing}")
    if frame["source_utc"].duplicated().any():
        raise ValueError("changepoint context requires one row per timestamp")
    result = pd.DataFrame(index=frame.index)
    per_signal = np.full((len(frame), len(input_columns)), np.nan, dtype=np.float32)
    for run in _continuous_runs(frame):
        for column, name in enumerate(input_columns):
            values = pd.to_numeric(frame.iloc[run][name], errors="coerce").to_numpy(float)
            per_signal[run, column] = _causal_scaled_scores(values, config=config)
    finite = np.isfinite(per_signal)
    count = finite.sum(axis=1)
    result["bocpd_context__mean_probability"] = np.divide(
        np.nansum(per_signal, axis=1), count, out=np.full(len(frame), np.nan, dtype=np.float32), where=count > 0
    ).astype(np.float32)
    max_score = np.max(np.where(finite, per_signal, -np.inf), axis=1)
    result["bocpd_context__max_probability"] = np.where(
        count > 0, max_score, np.nan
    ).astype(np.float32)
    result["bocpd_context__break_count_ge_0_10"] = np.where(
        count > 0, (per_signal >= 0.10).sum(axis=1), np.nan
    ).astype(np.float32)
    result["bocpd_context__break_count_ge_0_25"] = np.where(
        count > 0, (per_signal >= 0.25).sum(axis=1), np.nan
    ).astype(np.float32)
    return result, list(CHANGEPOINT_FEATURE_COLUMNS)
