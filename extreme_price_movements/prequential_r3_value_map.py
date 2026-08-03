"""Prior-resolved same-side R3 score-to-value map.

This is *not* the 21-day admission map.  It turns a strict-OOF R3 opportunity
score into prequential expected exact-net bps so the shared residual target has
compatible units.  Every row is mapped from labels whose availability time is
strictly before that row's decision time; rows at the same decision timestamp
are mapped together and cannot learn from one another.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


VALUE_MAP_SCHEMA = "prequential_same_side_r3_value_map_v1"
SCORE_SEMANTICS = "p_clear_minus_p_adverse"
OUTPUT_SEMANTICS = "prequential_base_expected_net_bps"


@dataclass(frozen=True)
class PrequentialR3ValueMapConfig:
    bins: int = 20
    min_global_rows: int = 32
    bin_shrink_rows: float = 64.0
    side: str = ""

    def validate(self) -> None:
        if int(self.bins) < 2:
            raise ValueError("R3 value map requires at least two fixed score bins")
        if int(self.min_global_rows) < 1 or float(self.bin_shrink_rows) <= 0.0:
            raise ValueError("R3 value-map support parameters must be positive")
        if str(self.side).lower() not in {"long", "short"}:
            raise ValueError("R3 value map must be fit independently for side=long or side=short")


def _utc(values: Sequence[Any], *, name: str) -> pd.Series:
    out = pd.to_datetime(pd.Series(values), utc=True, errors="coerce")
    if out.isna().any():
        raise ValueError(f"{name} contains missing/non-UTC-convertible timestamps")
    return out


def r3_opportunity_score(
    *,
    p_clear: Sequence[float] | None = None,
    p_adverse: Sequence[float] | None = None,
    p_weak: Sequence[float] | None = None,
    score: Sequence[float] | None = None,
) -> np.ndarray:
    """Validate R3 probabilities or a declared P(clear)-P(adverse) score."""
    if score is not None:
        if any(value is not None for value in (p_clear, p_adverse, p_weak)):
            raise ValueError("supply either R3 probabilities or scalar score, not both")
        out = np.asarray(score, dtype=np.float64).reshape(-1)
        if not np.isfinite(out).all() or np.any(out < -1.0) or np.any(out > 1.0):
            raise ValueError("R3 scalar score must be finite P(clear)-P(adverse) in [-1, 1]")
        return out
    if p_clear is None or p_adverse is None or p_weak is None:
        raise ValueError("R3 value map requires p_clear, p_adverse, p_weak or scalar score")
    clear = np.asarray(p_clear, dtype=np.float64).reshape(-1)
    adverse = np.asarray(p_adverse, dtype=np.float64).reshape(-1)
    weak = np.asarray(p_weak, dtype=np.float64).reshape(-1)
    if not (len(clear) == len(adverse) == len(weak)):
        raise ValueError("R3 probability arrays must be aligned")
    probabilities = np.column_stack([adverse, weak, clear])
    if not np.isfinite(probabilities).all() or (probabilities < -1e-6).any():
        raise ValueError("R3 probabilities must be finite and non-negative")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("R3 probabilities must sum to one")
    return clear - adverse


def prequential_same_side_r3_value_map(
    *,
    exact_net_bps: Sequence[float],
    decision_timestamps: Sequence[Any],
    label_available_timestamps: Sequence[Any],
    side: str,
    p_clear: Sequence[float] | None = None,
    p_adverse: Sequence[float] | None = None,
    p_weak: Sequence[float] | None = None,
    score: Sequence[float] | None = None,
    config: PrequentialR3ValueMapConfig | None = None,
) -> tuple[np.ndarray, pd.DataFrame, Mapping[str, Any]]:
    """Return causal expected-net bps, a row audit, and provenance.

    Bins are fixed over the known R3 domain [-1, 1], not fitted from future
    scores.  The global and bin means at each decision timestamp use only
    ``label_available_ts < decision_ts``.  Bin estimates shrink to the same
    prior-resolved global mean; before sufficient support the explicit neutral
    fallback is 0 bps.
    """
    cfg = config or PrequentialR3ValueMapConfig(side=side)
    if str(cfg.side).lower() != str(side).lower():
        raise ValueError("R3 value-map config side does not match the supplied side")
    cfg.validate()
    raw = r3_opportunity_score(
        p_clear=p_clear, p_adverse=p_adverse, p_weak=p_weak, score=score
    )
    net = np.asarray(exact_net_bps, dtype=np.float64).reshape(-1)
    decision = _utc(decision_timestamps, name="decision_timestamps")
    available = _utc(label_available_timestamps, name="label_available_timestamps")
    n = len(raw)
    if len(net) != n or len(decision) != n or len(available) != n:
        raise ValueError("R3 value-map inputs must be row-aligned")
    if not np.isfinite(net).all() or (available <= decision).any():
        raise ValueError("exact net must be finite and labels must resolve strictly after decision")

    # Fixed score-domain buckets: 0 through bins-1.  The upper edge is clipped
    # to the final bucket so score=1 remains valid.
    bucket = np.minimum(
        int(cfg.bins) - 1,
        np.floor((np.clip(raw, -1.0, 1.0) + 1.0) * int(cfg.bins) / 2.0).astype(int),
    )
    output = np.zeros(n, dtype=np.float32)
    support = np.zeros(n, dtype=np.int32)
    global_support = np.zeros(n, dtype=np.int32)
    fallback = np.empty(n, dtype=object)
    max_resolution = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
    # Process decision groups and newly-resolved labels as two independently
    # sorted event streams.  The previous implementation rebuilt
    # ``available < cutoff`` (and every score-bin mask) across the complete
    # side population for every decision timestamp.  That was semantically
    # correct but quadratic in production, where one side contains millions
    # of rows.  Running sufficient statistics preserve the exact strict
    # ``label_available_ts < decision_ts`` boundary in O(n log n) time.
    decision_values = decision.to_numpy(dtype="datetime64[ns]")
    available_values = available.to_numpy(dtype="datetime64[ns]")
    order = np.argsort(decision_values, kind="stable")
    ordered_decision = decision_values[order]
    availability_order = np.argsort(available_values, kind="stable")
    bin_count = np.zeros(int(cfg.bins), dtype=np.int64)
    bin_sum = np.zeros(int(cfg.bins), dtype=np.float64)
    resolved_count = 0
    resolved_sum = 0.0
    resolution_cursor = 0
    start = 0
    while start < n:
        stop = start + 1
        while stop < n and ordered_decision[stop] == ordered_decision[start]:
            stop += 1
        positions = order[start:stop]
        cutoff = ordered_decision[start]
        while (
            resolution_cursor < n
            and available_values[availability_order[resolution_cursor]] < cutoff
        ):
            resolved_position = int(availability_order[resolution_cursor])
            resolved_bucket = int(bucket[resolved_position])
            resolved_value = float(net[resolved_position])
            resolved_count += 1
            resolved_sum += resolved_value
            bin_count[resolved_bucket] += 1
            bin_sum[resolved_bucket] += resolved_value
            resolution_cursor += 1
        global_support[positions] = resolved_count
        if resolved_count < int(cfg.min_global_rows):
            fallback[positions] = "neutral_no_prior_resolved_support"
        else:
            global_mean = resolved_sum / resolved_count
            # The availability stream is sorted, so the most recently added
            # event is the exact maximum strictly before this decision group.
            max_resolution[positions] = available_values[
                availability_order[resolution_cursor - 1]
            ]
            for value in np.unique(bucket[positions]):
                target_positions = positions[bucket[positions] == value]
                count = int(bin_count[int(value)])
                support[target_positions] = count
                if count == 0:
                    output[target_positions] = global_mean
                    fallback[target_positions] = "global_prior_fallback_empty_bin"
                else:
                    bin_mean = float(bin_sum[int(value)] / count)
                    shrink = float(cfg.bin_shrink_rows)
                    output[target_positions] = (count * bin_mean + shrink * global_mean) / (count + shrink)
                    fallback[target_positions] = "shrunk_bin_prior_resolved"
        start = stop
    audit = pd.DataFrame(
        {
            "r3_opportunity_score": raw.astype(np.float32),
            "r3_score_bin": bucket.astype(np.int16),
            "prequential_base_expected_net_bps": output,
            "prior_resolved_global_support": global_support,
            "prior_resolved_bin_support": support,
            "value_map_fallback": fallback,
            "value_map_max_label_available_ts": pd.to_datetime(max_resolution, utc=True),
            "side": str(side).lower(),
        }
    )
    provenance: Mapping[str, Any] = {
        "schema": VALUE_MAP_SCHEMA,
        "side": str(side).lower(),
        "strict_oof": True,
        "units": "bps",
        "score_semantics": OUTPUT_SEMANTICS,
        "input_score_semantics": SCORE_SEMANTICS,
        "is_21_day_admission_map": False,
        "prior_resolution_rule": "label_available_ts < decision_ts",
        "bins": int(cfg.bins),
        "min_global_rows": int(cfg.min_global_rows),
        "bin_shrink_rows": float(cfg.bin_shrink_rows),
    }
    return output, audit, provenance


__all__ = [
    "OUTPUT_SEMANTICS",
    "PrequentialR3ValueMapConfig",
    "SCORE_SEMANTICS",
    "VALUE_MAP_SCHEMA",
    "prequential_same_side_r3_value_map",
    "r3_opportunity_score",
]
