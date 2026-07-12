"""Inference-safe cross-sectional geometry for meta residual calibration.

All features use only the current candidate batch and lagged candidate
membership. Realized returns and execution outcomes are never inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

DEFAULT_RELATIVE_FEATURES: tuple[str, ...] = (
    "asset_minus_mkt_oi_chg_1h_rz",
    "asset_minus_mkt_oi_chg_4h_rz",
    "asset_minus_mkt_oi_drawdown_24h",
    "asset_minus_mkt_oi_recovery_fraction_24h",
    "asset_minus_mkt_price_recovery_fraction_24h",
    "asset_minus_mkt_long_flush_intensity_4h",
    "asset_minus_mkt_short_cover_intensity_1h",
    "asset_mkt_liquidation_phase_divergence",
    "asset_mkt_exhaustion_phase_divergence",
    "ret48h_bench_resid",
    "rv_rel_universe",
    "carry_adj_ret_self_z_10h",
)

GEOMETRY_PREFIX = "meta_xsgeom_"


def _safe_name(name: str) -> str:
    return str(name).replace("__", "_").strip("_")


def geometry_feature_names(
    relative_features: Sequence[str] = DEFAULT_RELATIVE_FEATURES,
) -> list[str]:
    names = [
        f"{GEOMETRY_PREFIX}candidate_count_log1p",
        f"{GEOMETRY_PREFIX}score_std",
        f"{GEOMETRY_PREFIX}score_iqr",
        f"{GEOMETRY_PREFIX}top10_score_margin",
        f"{GEOMETRY_PREFIX}top10_score_mass_share",
        f"{GEOMETRY_PREFIX}top10_score_hhi",
        f"{GEOMETRY_PREFIX}top10_long_share",
        f"{GEOMETRY_PREFIX}top10_archetype_hhi",
        f"{GEOMETRY_PREFIX}top10_archetype_max_share",
        f"{GEOMETRY_PREFIX}top10_turnover_1h",
        f"{GEOMETRY_PREFIX}top10_turnover_4h",
        f"{GEOMETRY_PREFIX}local_support_share",
        f"{GEOMETRY_PREFIX}local_score_mean_delta",
        f"{GEOMETRY_PREFIX}local_score_rank",
    ]
    for source in relative_features:
        safe = _safe_name(source)
        names.extend(
            (
                f"{GEOMETRY_PREFIX}{safe}_iqr",
                f"{GEOMETRY_PREFIX}{safe}_top10_delta",
            )
        )
    return names


def _jaccard_turnover(current: frozenset[str], previous: frozenset[str]) -> float:
    union = current | previous
    if not union:
        return 0.0
    return float(1.0 - len(current & previous) / len(union))


def _membership_turnover(
    timestamps: pd.Series,
    symbols: pd.Series,
    selected: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    membership = (
        pd.DataFrame(
            {
                "timestamp": timestamps,
                "symbol": symbols.astype(str),
                "selected": selected.fillna(False).astype(bool),
            }
        )
        .loc[lambda values: values["selected"]]
        .groupby("timestamp", sort=True)["symbol"]
        .agg(lambda values: frozenset(values))
    )
    all_timestamps = pd.DatetimeIndex(
        pd.Series(timestamps).dropna().unique()
    ).sort_values()
    membership = membership.reindex(all_timestamps, fill_value=frozenset())
    lookup = membership.to_dict()
    turnover_1h = pd.Series(
        {
            timestamp: _jaccard_turnover(
                members, lookup.get(timestamp - pd.Timedelta(hours=1), frozenset())
            )
            for timestamp, members in membership.items()
        },
        dtype=np.float32,
    )
    turnover_4h = pd.Series(
        {
            timestamp: _jaccard_turnover(
                members, lookup.get(timestamp - pd.Timedelta(hours=4), frozenset())
            )
            for timestamp, members in membership.items()
        },
        dtype=np.float32,
    )
    return turnover_1h, turnover_4h


def materialize_cross_sectional_geometry(
    frame: pd.DataFrame,
    *,
    score_col: str,
    timestamp_col: str = "__ts__",
    symbol_col: str = "__symbol__",
    side_col: str = "side_name",
    archetype_col: str = "archetype_policy_key",
    relative_features: Sequence[str] = DEFAULT_RELATIVE_FEATURES,
) -> pd.DataFrame:
    """Return row-aligned, causal cross-sectional geometry features."""

    required = (score_col, timestamp_col, symbol_col, side_col, archetype_col)
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"Cross-sectional geometry is missing columns: {missing}")
    timestamp = pd.to_datetime(frame[timestamp_col], utc=True, errors="coerce")
    score = pd.to_numeric(frame[score_col], errors="coerce").astype(np.float32)
    rank = score.groupby(timestamp, sort=False).rank(method="average", pct=True)
    top10 = rank.ge(0.90)
    output = pd.DataFrame(index=frame.index)

    count = score.groupby(timestamp, sort=False).transform("count").clip(lower=1)
    score_mean = score.groupby(timestamp, sort=False).transform("mean")
    score_std = score.groupby(timestamp, sort=False).transform("std").fillna(0.0)
    q25 = score.groupby(timestamp, sort=False).transform("quantile", q=0.25)
    q75 = score.groupby(timestamp, sort=False).transform("quantile", q=0.75)
    q90 = score.groupby(timestamp, sort=False).transform("quantile", q=0.90)
    top10_score = score.where(top10)
    top10_mean = top10_score.groupby(timestamp, sort=False).transform("mean")
    positive_mass = (
        score - score.groupby(timestamp, sort=False).transform("min")
    ).clip(lower=0.0) + 1e-6
    total_mass = positive_mass.groupby(timestamp, sort=False).transform("sum")
    top_mass = (
        positive_mass.where(top10, 0.0).groupby(timestamp, sort=False).transform("sum")
    )
    top_weight = positive_mass.where(top10, 0.0) / top_mass.replace(0.0, np.nan)
    top_hhi = top_weight.pow(2).groupby(timestamp, sort=False).transform("sum")

    side_long = frame[side_col].astype(str).str.lower().eq("long").astype(np.float32)
    top_count = top10.groupby(timestamp, sort=False).transform("sum").clip(lower=1)
    top_long_share = (
        side_long.where(top10, 0.0).groupby(timestamp, sort=False).transform("sum")
        / top_count
    )

    top_arch = pd.DataFrame(
        {
            "timestamp": timestamp,
            "archetype": frame[archetype_col].astype(str),
            "top10": top10,
        },
        index=frame.index,
    )
    arch_counts = (
        top_arch.loc[top10]
        .groupby(["timestamp", "archetype"], observed=True, sort=False)
        .size()
        .rename("count")
        .reset_index()
    )
    if arch_counts.empty:
        arch_hhi = pd.Series(0.0, index=pd.DatetimeIndex(timestamp.dropna().unique()))
        arch_max = arch_hhi.copy()
    else:
        arch_counts["total"] = arch_counts.groupby("timestamp", sort=False)[
            "count"
        ].transform("sum")
        arch_counts["share"] = arch_counts["count"] / arch_counts["total"].clip(lower=1)
        arch_hhi = arch_counts.groupby("timestamp", sort=False)["share"].apply(
            lambda x: float(np.square(x).sum())
        )
        arch_max = arch_counts.groupby("timestamp", sort=False)["share"].max()

    turnover_1h, turnover_4h = _membership_turnover(timestamp, frame[symbol_col], top10)
    local_key = (
        timestamp.astype(str)
        + "|"
        + frame[side_col].astype(str).str.lower()
        + "|"
        + frame[archetype_col].astype(str)
    )
    local_count = score.groupby(local_key, sort=False).transform("count")
    local_mean = score.groupby(local_key, sort=False).transform("mean")
    local_rank = score.groupby(local_key, sort=False).rank(method="average", pct=True)

    output[f"{GEOMETRY_PREFIX}candidate_count_log1p"] = np.log1p(count).astype(
        np.float32
    )
    output[f"{GEOMETRY_PREFIX}score_std"] = score_std.astype(np.float32)
    output[f"{GEOMETRY_PREFIX}score_iqr"] = (q75 - q25).fillna(0.0).astype(np.float32)
    output[f"{GEOMETRY_PREFIX}top10_score_margin"] = (
        (top10_mean - q90).fillna(0.0).astype(np.float32)
    )
    output[f"{GEOMETRY_PREFIX}top10_score_mass_share"] = (
        (top_mass / total_mass.clip(lower=1e-6)).fillna(0.0).astype(np.float32)
    )
    output[f"{GEOMETRY_PREFIX}top10_score_hhi"] = top_hhi.fillna(0.0).astype(np.float32)
    output[f"{GEOMETRY_PREFIX}top10_long_share"] = top_long_share.fillna(0.0).astype(
        np.float32
    )
    output[f"{GEOMETRY_PREFIX}top10_archetype_hhi"] = (
        timestamp.map(arch_hhi).fillna(0.0).astype(np.float32)
    )
    output[f"{GEOMETRY_PREFIX}top10_archetype_max_share"] = (
        timestamp.map(arch_max).fillna(0.0).astype(np.float32)
    )
    output[f"{GEOMETRY_PREFIX}top10_turnover_1h"] = (
        timestamp.map(turnover_1h).fillna(1.0).astype(np.float32)
    )
    output[f"{GEOMETRY_PREFIX}top10_turnover_4h"] = (
        timestamp.map(turnover_4h).fillna(1.0).astype(np.float32)
    )
    output[f"{GEOMETRY_PREFIX}local_support_share"] = (
        (local_count / count).fillna(0.0).astype(np.float32)
    )
    output[f"{GEOMETRY_PREFIX}local_score_mean_delta"] = (
        (local_mean - score_mean).fillna(0.0).astype(np.float32)
    )
    output[f"{GEOMETRY_PREFIX}local_score_rank"] = local_rank.fillna(0.5).astype(
        np.float32
    )

    for source in relative_features:
        if source not in frame.columns:
            continue
        values = pd.to_numeric(frame[source], errors="coerce").astype(np.float32)
        median = values.groupby(timestamp, sort=False).transform("median")
        lower = values.groupby(timestamp, sort=False).transform("quantile", q=0.25)
        upper = values.groupby(timestamp, sort=False).transform("quantile", q=0.75)
        selected_median = (
            values.where(top10).groupby(timestamp, sort=False).transform("median")
        )
        safe = _safe_name(source)
        output[f"{GEOMETRY_PREFIX}{safe}_iqr"] = (
            (upper - lower).fillna(0.0).astype(np.float32)
        )
        output[f"{GEOMETRY_PREFIX}{safe}_top10_delta"] = (
            (selected_median - median).fillna(0.0).astype(np.float32)
        )

    return output.reindex(
        columns=geometry_feature_names(relative_features), fill_value=0.0
    ).astype(np.float32, copy=False)


@dataclass
class CrossSectionalGeometryState:
    """Minimal live state for lagged top-decile membership turnover."""

    top_memberships: dict[pd.Timestamp, frozenset[str]] = field(default_factory=dict)
    max_history_hours: int = 4

    def update(self, timestamp: pd.Timestamp, symbols: Sequence[str]) -> None:
        ts = pd.Timestamp(timestamp)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        self.top_memberships[ts] = frozenset(str(symbol) for symbol in symbols)
        cutoff = ts - pd.Timedelta(hours=self.max_history_hours)
        self.top_memberships = {
            key: value for key, value in self.top_memberships.items() if key >= cutoff
        }

    def turnover(
        self, timestamp: pd.Timestamp, symbols: Sequence[str], lag_hours: int
    ) -> float:
        ts = pd.Timestamp(timestamp)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return _jaccard_turnover(
            frozenset(str(symbol) for symbol in symbols),
            self.top_memberships.get(
                ts - pd.Timedelta(hours=int(lag_hours)), frozenset()
            ),
        )
