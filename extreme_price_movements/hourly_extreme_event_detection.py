"""Causal hourly state construction for extreme-period event research.

The event calendar is daily, but the observable state is hourly.  This module
therefore makes no claim about sub-hour onset timing: every score at ``t`` is
built from data available through ``t - 1h`` and can only anticipate the next
calendar-day onset at hourly resolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


HOURLY_FORBIDDEN_TOKENS = ("5m", "15m", "30m", "minute", "min_")

# Observable OHLCV/OI/funding/market-state variables.  The caller intersects
# this list with the frozen source schema, allowing earlier state artifacts to
# remain usable without silently falling back to outcome or model features.
HOURLY_EVENT_FEATURE_KEYS: tuple[str, ...] = (
    "mkt_ret_1h",
    "mkt_ret_4h",
    "mkt_rv_1h",
    "mkt_rv_4h",
    "mkt_rv_ratio_1h_24h",
    "mkt_atr_expansion_1h",
    "mkt_volume_z_24h",
    "market_breadth_chg_1h",
    "market_breadth_accel_1h",
    "market_breadth_recovery_from_6h_min",
    "negative_breadth_pct",
    "extreme_negative_breadth_pct",
    "downside_breadth_intensity",
    "mkt_oi_chg_1h",
    "mkt_oi_chg_4h",
    "mkt_oi_chg_accel_1h",
    "mkt_oi_flush_z_30d",
    "mkt_oi_flush_breadth_accel_1h",
    "mkt_oi_flush_breadth_recovery_4h",
    "mkt_median_oi_chg_1h_rz",
    "mkt_median_oi_chg_4h_rz",
    "mkt_pct_oi_chg_1h_rz_lt_minus1",
    "mkt_pct_oi_chg_4h_rz_lt_minus2",
    "mkt_funding_mean_z_30d",
    "mkt_funding_chg_1h",
    "mkt_funding_chg_4h",
    "mkt_funding_accel_4h",
    "mkt_funding_dispersion_z_30d",
    "funding_crowding_release_4h",
    "funding_positive_to_negative_intensity",
    "funding_negative_to_positive_intensity",
    "funding_deleveraging_divergence",
    "mkt_pct_price_down_oi_down_1h",
    "mkt_pct_price_up_oi_down_1h",
    "mkt_pct_price_up_oi_up_1h",
    "mkt_median_short_cover_intensity_1h",
    "short_covering_score_market",
    "flush_recovery_state",
    "post_flush_leverage_rebuild",
    "market_pc1_variance_share_12h",
    "market_pc1_variance_share_24h",
    "market_pc1_variance_share_chg_4h",
    "market_downside_pairwise_corr_24h",
    "market_downside_corr_minus_unconditional_corr_24h",
    "mkt_systemic_deleveraging_score",
    "mkt_flush_exhaustion_score",
    "mkt_leverage_rebuild_score",
    "btc_resilience_alt_weakness",
    "median_alt_minus_btc",
    "breadth_dispersion",
    "peer_decoupling_acceleration",
)

# We deliberately derive transition features only on market-wide state columns.
# The raw values are shifted one full hourly bar before all transforms.
TRANSITION_STEMS: tuple[str, ...] = (
    "mkt_oi_chg_1h",
    "mkt_funding_mean_z_30d",
    "market_breadth_chg_1h",
    "market_downside_pairwise_corr_24h",
    "market_pc1_variance_share_24h",
    "btc_resilience_alt_weakness",
    "short_covering_score_market",
    "flush_recovery_state",
    "post_flush_leverage_rebuild",
)

MECHANISM_MEMORY_HALF_LIFE_HOURS: dict[str, float] = {
    "liquidation_pressure": 6.0,
    "recovery_short_covering": 8.0,
    "funding_transition": 12.0,
    "correlation_fragmentation": 24.0,
    "volatility_compression_transition": 18.0,
    "asset_market_divergence": 24.0,
}


@dataclass(frozen=True)
class HourlyEventConfig:
    """Time and support constraints for the hourly-only detector."""

    decision_lag_hours: int = 1
    lead_hours: int = 12
    state_hours: int = 24
    embargo_hours: int = 36
    control_ratio: int = 4
    max_features: int = 32


def assert_hourly_only(columns: Iterable[str]) -> None:
    """Reject accidental sub-hour inputs rather than silently accepting them."""

    forbidden = sorted(
        name
        for name in columns
        if any(token in str(name).lower() for token in HOURLY_FORBIDDEN_TOKENS)
    )
    if forbidden:
        raise ValueError(f"Hourly detector cannot use sub-hour columns: {forbidden}")


def available_hourly_features(columns: Iterable[str]) -> list[str]:
    """Return supported observable state features, excluding sub-hour inputs."""

    available = set(map(str, columns))
    selected = [name for name in HOURLY_EVENT_FEATURE_KEYS if name in available]
    assert_hourly_only(selected)
    return selected


def _robust_cross_sectional_median(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    numeric = frame.loc[:, ["__ts__", *columns]].copy()
    for name in columns:
        numeric[name] = pd.to_numeric(numeric[name], errors="coerce").astype(np.float32)
    return (
        numeric.groupby("__ts__", as_index=False, sort=True)[list(columns)]
        .median()
        .sort_values("__ts__", kind="stable")
        .reset_index(drop=True)
    )


def build_hourly_market_state(
    rows: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    config: HourlyEventConfig = HourlyEventConfig(),
) -> pd.DataFrame:
    """Collapse an hourly candidate state to a market state with causal lags.

    Candidate rows can contain several symbols, sides, and archetypes at one
    timestamp.  A robust cross-sectional median prevents the selected mix from
    creating a synthetic extreme.  The full state is lagged by one hourly bar,
    so neither a partially formed current bar nor an outcome enters the score.
    """

    required = {"__ts__"}
    missing = required.difference(rows.columns)
    if missing:
        raise KeyError(f"Hourly state missing required columns: {sorted(missing)}")
    features = list(dict.fromkeys(map(str, feature_columns)))
    assert_hourly_only(features)
    if not features:
        raise ValueError("No hourly observable features available")

    state = _robust_cross_sectional_median(rows, features)
    state["__ts__"] = pd.to_datetime(state["__ts__"], utc=True, errors="coerce")
    state = state.loc[state["__ts__"].notna()].sort_values("__ts__", kind="stable").reset_index(drop=True)
    expected = pd.date_range(state["__ts__"].min(), state["__ts__"].max(), freq="h", tz="UTC")
    state = state.set_index("__ts__").reindex(expected)
    state.index.name = "__ts__"
    state["hourly_state_coverage"] = state[features].notna().mean(axis=1).astype(np.float32)
    # Forward filling would hide data loss, therefore retain missing values and
    # let each fold's train-only imputer resolve them.
    state[features] = state[features].shift(max(int(config.decision_lag_hours), 1))

    transition_features: list[str] = []
    change_components: list[pd.Series] = []
    for stem in TRANSITION_STEMS:
        if stem not in state.columns:
            continue
        values = state[stem].astype(np.float32)
        delta = values.diff(1)
        state[f"evt_{stem}__delta_1h"] = delta.astype(np.float32)
        state[f"evt_{stem}__accel_1h"] = delta.diff(1).astype(np.float32)
        state[f"evt_{stem}__chg_24h"] = values.diff(24).astype(np.float32)
        state[f"evt_{stem}__sign_flip_1h"] = (
            np.sign(values).ne(np.sign(values.shift(1))).astype(np.float32)
        )
        transition_features.extend(
            [
                f"evt_{stem}__delta_1h",
                f"evt_{stem}__accel_1h",
                f"evt_{stem}__chg_24h",
                f"evt_{stem}__sign_flip_1h",
            ]
        )
        # Online, robust change-point proxy.  The current value is compared
        # only with the preceding 168 hourly states; it is never compared with
        # future observations or a fitted OOS distribution.  This gives the
        # event model a compact transition signal without a costly matrix
        # profile over a large multi-asset panel.
        history = values.shift(1)
        baseline = history.rolling(168, min_periods=24).median()
        mad = (history - baseline).abs().rolling(168, min_periods=24).median() * 1.4826
        change_components.append(((values - baseline) / (mad + 1e-4)).clip(-8.0, 8.0).abs())
    if change_components:
        state["evt_causal_change_score"] = pd.concat(change_components, axis=1).mean(axis=1).astype(np.float32)
        transition_features.append("evt_causal_change_score")
    state = state.reset_index()
    state["day"] = state["__ts__"].dt.floor("D")
    state.attrs["observable_features"] = features
    state.attrs["transition_features"] = transition_features
    state.attrs["causal_contract"] = (
        f"cross-sectional hourly median, then {config.decision_lag_hours}h lag; "
        "transition transforms are derived after the lag"
    )
    return state


def calendar_hourly_targets(
    hourly: pd.DataFrame,
    calendar: pd.DataFrame,
    taxonomy: pd.DataFrame,
    *,
    config: HourlyEventConfig = HourlyEventConfig(),
) -> pd.DataFrame:
    """Create global hourly event labels from daily side/archetype event cells.

    These are training/evaluation labels only.  Their use is restricted to
    target construction; no calendar or realized outcome field is copied into
    inference features.
    """

    result = hourly.loc[:, ["__ts__", "day"]].copy()
    calendar = calendar.copy()
    calendar["day"] = pd.to_datetime(calendar["day"], utc=True).dt.floor("D")
    event_column = "adverse_calendar_cell" if "adverse_calendar_cell" in calendar.columns else "adverse_event_rows"
    event_days = (
        calendar.assign(_event=pd.to_numeric(calendar[event_column], errors="coerce").fillna(0).gt(0))
        .groupby("day", as_index=False) ["_event"].max()
        .rename(columns={"_event": "event_state"})
    )
    result = result.merge(event_days, on="day", how="left")
    result["event_state"] = pd.to_numeric(result["event_state"], errors="coerce").fillna(0).astype(np.int8)
    # One onset per contiguous event block.  A daily calendar becomes 24
    # hourly state rows, but it must not turn one episode into 24 positives.
    prior_hour = result["event_state"].shift(1, fill_value=0)
    result["event_onset"] = (result["event_state"].eq(1) & prior_hour.eq(0)).astype(np.int8)

    target = np.zeros(len(result), dtype=np.int8)
    onset = result["event_onset"].to_numpy(np.int8)
    for offset in range(1, max(int(config.lead_hours), 1) + 1):
        target[:-offset] = np.maximum(target[:-offset], onset[offset:])
    # The original name remains for compatibility with the first benchmark.
    # The phase names make it explicit that this is a pre-onset target, not an
    # active-stress label.
    result["event_onset_next_window"] = target
    result["event_pre_onset_next_window"] = target
    result["event_active_stress"] = result["event_state"].astype(np.int8)

    taxonomy = taxonomy.copy()
    taxonomy["event_start"] = pd.to_datetime(taxonomy["event_start"], utc=True).dt.floor("D")
    mechanisms = sorted(
        name
        for name in taxonomy.get("onset_primary_mechanism", pd.Series(dtype=str)).dropna().astype(str).unique()
        if name and name != "unavailable"
    )
    for mechanism in mechanisms:
        mechanism_rows = taxonomy.loc[taxonomy["onset_primary_mechanism"].eq(mechanism)].copy()
        starts = pd.to_datetime(mechanism_rows["event_start"], utc=True).dt.floor("h").drop_duplicates()
        onset = result["__ts__"].isin(starts).to_numpy(np.int8)
        pre_onset = np.zeros(len(result), dtype=np.int8)
        for offset in range(1, max(int(config.lead_hours), 1) + 1):
            pre_onset[:-offset] = np.maximum(pre_onset[:-offset], onset[offset:])
        active = np.zeros(len(result), dtype=np.int8)
        recovery = np.zeros(len(result), dtype=np.int8)
        for row in mechanism_rows.itertuples(index=False):
            start = pd.Timestamp(row.event_start)
            end = pd.Timestamp(row.event_end)
            start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
            end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
            # Calendar dates are inclusive; a one-day event ending at midnight
            # therefore owns the complete end date through the next midnight.
            active |= result["__ts__"].ge(start).to_numpy(np.int8) & result["__ts__"].lt(end + pd.Timedelta(days=1)).to_numpy(np.int8)
            recovery |= result["__ts__"].ge(end + pd.Timedelta(days=1)).to_numpy(np.int8) & result["__ts__"].lt(end + pd.Timedelta(days=1, hours=config.lead_hours)).to_numpy(np.int8)
        result[f"mechanism__{mechanism}"] = active.astype(np.int8)
        result[f"mechanism__{mechanism}__onset"] = onset
        result[f"mechanism__{mechanism}__pre_onset_next_window"] = pre_onset
        result[f"mechanism__{mechanism}__active_stress"] = active.astype(np.int8)
        result[f"mechanism__{mechanism}__recovery"] = recovery.astype(np.int8)
    return result


def matched_control_weights(
    frame: pd.DataFrame,
    *,
    target_column: str,
    volatility_column: str,
    control_ratio: int,
    seed: int,
) -> np.ndarray:
    """Match normal controls by hour, weekday, and train-only volatility decile."""

    target = pd.to_numeric(frame[target_column], errors="coerce").fillna(0).to_numpy(np.int8)
    weights = np.zeros(len(frame), dtype=np.float32)
    positive = np.flatnonzero(target > 0)
    if not len(positive):
        return weights
    values = pd.to_numeric(frame[volatility_column], errors="coerce") if volatility_column in frame else pd.Series(np.nan, index=frame.index)
    finite = values.dropna()
    if len(finite) >= 20:
        edges = np.unique(np.nanquantile(finite.to_numpy(np.float32), np.linspace(0.0, 1.0, 11)))
        bucket = np.digitize(values.fillna(float(finite.median())), edges[1:-1], right=True)
    else:
        bucket = np.zeros(len(frame), dtype=np.int8)
    timestamp = pd.to_datetime(frame["__ts__"], utc=True)
    strata = pd.DataFrame({"hour": timestamp.dt.hour, "dow": timestamp.dt.dayofweek, "vol": bucket})
    normal = target == 0
    rng = np.random.default_rng(seed)
    chosen: list[np.ndarray] = []
    for idx in positive:
        same = np.flatnonzero(
            normal
            & strata["hour"].eq(strata.iloc[idx]["hour"]).to_numpy()
            & strata["dow"].eq(strata.iloc[idx]["dow"]).to_numpy()
            & strata["vol"].eq(strata.iloc[idx]["vol"]).to_numpy()
        )
        if not len(same):
            same = np.flatnonzero(normal & strata["hour"].eq(strata.iloc[idx]["hour"]).to_numpy())
        if len(same):
            chosen.append(rng.choice(same, size=min(int(control_ratio), len(same)), replace=False))
    weights[positive] = 1.0
    if chosen:
        weights[np.concatenate(chosen)] = 1.0
    return weights


def causal_episode_memory(
    event_score: np.ndarray,
    mechanism_scores: dict[str, np.ndarray],
    *,
    threshold: float,
) -> pd.DataFrame:
    """Persist predicted state without using labels or realized outcomes."""

    output: dict[str, np.ndarray] = {}
    trigger = np.nan_to_num(event_score, nan=0.0) >= float(threshold)
    for mechanism, scores in mechanism_scores.items():
        half_life = MECHANISM_MEMORY_HALF_LIFE_HOURS.get(mechanism, 12.0)
        decay = float(np.exp(np.log(0.5) / max(half_life, 1.0)))
        memory = np.zeros(len(scores), dtype=np.float32)
        values = np.nan_to_num(scores, nan=0.0).astype(np.float32, copy=False)
        for index in range(1, len(memory)):
            impulse = values[index - 1] if trigger[index - 1] else 0.0
            memory[index] = max(memory[index - 1] * decay, impulse)
        output[f"event_memory__{mechanism}"] = memory
    return pd.DataFrame(output)
