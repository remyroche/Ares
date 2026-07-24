"""P90-spread, trailing-aware base-label target variants.

The S59 materialized labels already contain causal trailing outcomes.  This
module only *reprices* those outcomes: it reconstructs gross return from the
stored net return, replaces the old fixed cost once, and produces bounded soft
targets.  It supports both a strict causal estimator and a static asset-liquidity
proxy used for label research where historical spread snapshots are incomplete.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd


KEY_COLUMNS = ("__ts__", "__symbol__", "side_name")


@dataclass(frozen=True)
class CausalSpreadP90Spec:
    lookback_days: int = 28
    quantile: float = 0.90
    min_observations: int = 48
    min_distinct_days: int = 7
    fee_round_trip: float = 0.003


@dataclass(frozen=True)
class TrailingCostTargetSpec:
    """Side-specific bounded target parameters.

    ``blend`` is deliberately limited to a modest correction: the incumbent
    materialized target remains the source of trailing/path ordering while the
    new net economics changes its strength.
    """

    margin: float
    temperature: float
    blend: float
    activation_bonus: float
    slow_timeout_penalty: float
    adverse_path_penalty: float


DEFAULT_TARGET_SPECS: Mapping[str, TrailingCostTargetSpec] = {
    "long": TrailingCostTargetSpec(
        margin=0.0,
        temperature=0.012,
        blend=0.35,
        activation_bonus=0.08,
        slow_timeout_penalty=0.12,
        adverse_path_penalty=0.10,
    ),
    "short": TrailingCostTargetSpec(
        margin=0.0,
        temperature=0.010,
        blend=0.30,
        activation_bonus=0.06,
        slow_timeout_penalty=0.14,
        adverse_path_penalty=0.12,
    ),
}


def normalize_side(values: pd.Series | np.ndarray) -> pd.Series:
    raw = pd.Series(values, copy=False)
    numeric = pd.to_numeric(raw, errors="coerce")
    text = raw.astype(str).str.lower()
    return pd.Series(
        np.where(text.str.contains("short", regex=False) | numeric.lt(0.0).fillna(False), "short", "long"),
        index=raw.index,
        dtype="string",
    )


def causal_p90_spread_cost(
    rows: pd.DataFrame,
    spread_history: pd.DataFrame,
    *,
    spec: CausalSpreadP90Spec = CausalSpreadP90Spec(),
) -> pd.DataFrame:
    """Return a strictly prior, per-symbol p90 full-spread cost estimate.

    The current history begins in June 2026.  Rows without enough *prior*
    observations remain unavailable rather than borrowing a later p90.  This
    makes it possible to compare a causal challenger against the full-history
    incumbent without claiming unavailable April/May spread evidence.
    """

    required_rows = {"__ts__", "__symbol__"}
    required_history = {"observed_ts", "symbol", "spread_bps"}
    missing_rows = sorted(required_rows.difference(rows.columns))
    missing_history = sorted(required_history.difference(spread_history.columns))
    if missing_rows:
        raise ValueError(f"Cost target rows missing columns={missing_rows}")
    if missing_history:
        raise ValueError(f"Spread history missing columns={missing_history}")
    if not 0.0 < float(spec.quantile) < 1.0:
        raise ValueError("spread quantile must lie in (0, 1)")

    out = pd.DataFrame(index=rows.index)
    out["p90_spread_bps"] = np.nan
    out["p90_spread_observations"] = 0
    out["p90_spread_distinct_days"] = 0
    out["p90_spread_cost_available"] = False

    work = pd.DataFrame(
        {
            "_pos": np.arange(len(rows), dtype=np.int64),
            "_ts": pd.to_datetime(rows["__ts__"], utc=True, errors="coerce"),
            "_symbol": rows["__symbol__"].astype(str).to_numpy(copy=False),
        }
    )
    history = pd.DataFrame(
        {
            "_ts": pd.to_datetime(spread_history["observed_ts"], utc=True, errors="coerce"),
            "_symbol": spread_history["symbol"].astype(str).to_numpy(copy=False),
            "_spread": pd.to_numeric(spread_history["spread_bps"], errors="coerce"),
        }
    )
    history = history.loc[
        history["_ts"].notna() & history["_spread"].ge(0.0) & np.isfinite(history["_spread"])
    ].sort_values(["_symbol", "_ts"], kind="mergesort")
    lookback = pd.Timedelta(days=int(spec.lookback_days))
    p90 = np.full(len(rows), np.nan, dtype=np.float64)
    support = np.zeros(len(rows), dtype=np.int32)
    days = np.zeros(len(rows), dtype=np.int16)

    # Groups keep memory proportional to one asset's history. The target-HPO
    # labels are hourly, so a vectorized searchsorted slice is ample here.
    for symbol, group in work.groupby("_symbol", sort=False, observed=True):
        hist = history.loc[history["_symbol"].eq(str(symbol))]
        if hist.empty:
            continue
        hist_ts = hist["_ts"].astype("int64").to_numpy(copy=False)
        hist_spread = hist["_spread"].to_numpy(dtype=np.float64, copy=False)
        hist_days = hist["_ts"].dt.floor("D").astype("int64").to_numpy(copy=False)
        positions = group["_pos"].to_numpy(dtype=np.int64, copy=False)
        row_ts = group["_ts"].astype("int64").to_numpy(copy=False)
        for pos, ts_ns in zip(positions, row_ts, strict=False):
            if ts_ns == np.iinfo(np.int64).min:
                continue
            # Strict < decision time: a same-timestamp snapshot cannot affect
            # a label decision made at that timestamp.
            right = int(np.searchsorted(hist_ts, ts_ns, side="left"))
            left = int(np.searchsorted(hist_ts, ts_ns - lookback.value, side="left"))
            n = right - left
            if n <= 0:
                continue
            support[pos] = n
            distinct = int(np.unique(hist_days[left:right]).size)
            days[pos] = distinct
            if n >= int(spec.min_observations) and distinct >= int(spec.min_distinct_days):
                p90[pos] = float(np.quantile(hist_spread[left:right], float(spec.quantile)))

    out["p90_spread_bps"] = p90.astype(np.float32)
    out["p90_spread_observations"] = support
    out["p90_spread_distinct_days"] = days
    out["p90_spread_cost_available"] = np.isfinite(p90)
    out["p90_round_trip_cost"] = (
        float(spec.fee_round_trip) + p90 / 10_000.0
    ).astype(np.float32)
    return out


def pooled_asset_p90_spread_cost(
    rows: pd.DataFrame,
    spread_history: pd.DataFrame,
    *,
    spec: CausalSpreadP90Spec = CausalSpreadP90Spec(),
) -> pd.DataFrame:
    """Return one static p90 full-spread estimate per asset.

    This is intentionally *not* point-in-time. It is a stable liquidity proxy
    used to make historical label geometry cost-sensitive when the snapshot
    database started after the historical training period. The returned frame
    marks that distinction explicitly; it must not be represented as an
    untouched-OOS transaction-cost estimate.
    """

    required_rows = {"__symbol__"}
    required_history = {"symbol", "spread_bps"}
    missing_rows = sorted(required_rows.difference(rows.columns))
    missing_history = sorted(required_history.difference(spread_history.columns))
    if missing_rows:
        raise ValueError(f"Cost target rows missing columns={missing_rows}")
    if missing_history:
        raise ValueError(f"Spread history missing columns={missing_history}")
    valid = pd.DataFrame(
        {
            "symbol": spread_history["symbol"].astype(str),
            "spread_bps": pd.to_numeric(spread_history["spread_bps"], errors="coerce"),
        }
    )
    valid = valid.loc[valid["spread_bps"].ge(0.0) & np.isfinite(valid["spread_bps"])]
    grouped = valid.groupby("symbol", observed=True)["spread_bps"]
    p90_by_symbol = grouped.quantile(float(spec.quantile))
    support_by_symbol = grouped.size()
    symbols = rows["__symbol__"].astype(str)
    p90 = symbols.map(p90_by_symbol).to_numpy(dtype=np.float64)
    support = symbols.map(support_by_symbol).fillna(0).to_numpy(dtype=np.int32)
    available = np.isfinite(p90) & (support >= int(spec.min_observations))
    p90[~available] = np.nan
    out = pd.DataFrame(index=rows.index)
    out["p90_spread_bps"] = p90.astype(np.float32)
    out["p90_spread_observations"] = support
    out["p90_spread_distinct_days"] = np.int16(-1)
    out["p90_spread_cost_available"] = available
    out["p90_round_trip_cost"] = (
        float(spec.fee_round_trip) + p90 / 10_000.0
    ).astype(np.float32)
    return out


def build_trailing_cost_targets(
    rows: pd.DataFrame,
    cost: pd.DataFrame,
    *,
    specs: Mapping[str, TrailingCostTargetSpec] = DEFAULT_TARGET_SPECS,
) -> pd.DataFrame:
    """Build a cost-aware soft target while preserving the incumbent fallback.

    Old 1% cost is removed by reconstructing gross from the materialized
    capture net. The new p90 full spread and 30bp fee are then charged once.
    Rows without causal cost support keep the incumbent target and are marked
    so promotion metrics can be restricted to the cost-observable population.
    """

    required = {"__first_touch_target_soft__", "__first_touch_capture_net__", "__first_touch_round_trip_cost__"}
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"Trailing cost target rows missing columns={missing}")
    if len(rows) != len(cost):
        raise ValueError("Trailing cost target rows and cost frame must align")

    incumbent = pd.to_numeric(rows["__first_touch_target_soft__"], errors="coerce").fillna(0.0).clip(0.0, 1.0).to_numpy(np.float64)
    old_net = pd.to_numeric(rows["__first_touch_capture_net__"], errors="coerce").to_numpy(np.float64)
    old_cost = pd.to_numeric(rows["__first_touch_round_trip_cost__"], errors="coerce").to_numpy(np.float64)
    gross = old_net + old_cost
    new_cost = pd.to_numeric(cost["p90_round_trip_cost"], errors="coerce").to_numpy(np.float64)
    observed = np.isfinite(gross) & np.isfinite(new_cost)
    net = gross - new_cost
    side_source = rows["side_name"] if "side_name" in rows else rows.get("side", pd.Series(1.0, index=rows.index))
    sides = normalize_side(side_source).to_numpy()

    activation = pd.to_numeric(rows.get("__trailing_profit_activated__", 0.0), errors="coerce").fillna(0.0).to_numpy(np.float64)
    timeout = pd.to_numeric(rows.get("__first_touch_timeout__", rows.get("__is_timeout__", 0.0)), errors="coerce").fillna(0.0).to_numpy(np.float64)
    mae = pd.to_numeric(rows.get("__first_touch_full_path_mae_norm__", 0.0), errors="coerce").fillna(0.0).to_numpy(np.float64)
    bars = pd.to_numeric(rows.get("__trailing_profit_activation_bar__", 0.0), errors="coerce").fillna(0.0).to_numpy(np.float64)

    target = incumbent.copy()
    economic = np.full(len(rows), np.nan, dtype=np.float64)
    multiplier = np.ones(len(rows), dtype=np.float64)
    for side in ("long", "short"):
        idx = np.flatnonzero((sides == side) & observed)
        if not len(idx):
            continue
        spec = specs[side]
        economic[idx] = 1.0 / (1.0 + np.exp(-np.clip((net[idx] - spec.margin) / max(spec.temperature, 1e-6), -50.0, 50.0)))
        slow = np.clip(bars[idx] / (24.0 if side == "long" else 16.0), 0.0, 1.0)
        multiplier[idx] = np.clip(
            1.0
            + spec.activation_bonus * np.clip(activation[idx], 0.0, 1.0)
            - spec.slow_timeout_penalty * np.clip(timeout[idx], 0.0, 1.0) * slow
            - spec.adverse_path_penalty * np.clip(mae[idx] - 1.0, 0.0, 2.0) / 2.0,
            0.55,
            1.10,
        )
        challenger = np.clip(economic[idx] * multiplier[idx], 0.0, 1.0)
        target[idx] = np.clip((1.0 - spec.blend) * incumbent[idx] + spec.blend * challenger, 0.0, 1.0)

    return pd.DataFrame(
        {
            "target_soft_p90_trailing_blend": target.astype(np.float32),
            "target_hard_p90_trailing_blend": (target >= 0.5).astype(np.float32),
            "__p90_trailing_target_soft__": target.astype(np.float32),
            "__p90_trailing_target_hard__": (target >= 0.5).astype(np.float32),
            "target_soft_incumbent": incumbent.astype(np.float32),
            "capture_gross_reconstructed": gross.astype(np.float32),
            "capture_net_p90_spread_fee30bps": net.astype(np.float32),
            # Backward-compatible alias for readers of older target artifacts.
            # Its name describes the legacy schema, not the active fee contract.
            "capture_net_p90_spread_fee15bps": net.astype(np.float32),
            "trailing_cost_economic_component": economic.astype(np.float32),
            "trailing_cost_path_multiplier": multiplier.astype(np.float32),
            "p90_cost_observed": observed.astype(np.float32),
        },
        index=rows.index,
    )


def target_contract_manifest(
    *,
    cost_spec: CausalSpreadP90Spec,
    target_specs: Mapping[str, TrailingCostTargetSpec],
    rows: int,
    observed_rows: int,
    cost_estimator: str = "causal_rolling_p90",
) -> dict[str, Any]:
    estimator = str(cost_estimator)
    is_causal = estimator == "causal_rolling_p90"
    return {
        "schema": "p90_trailing_target_v1",
        "cost": {
            **asdict(cost_spec),
            "estimator": estimator,
            "formula": (
                "gross_trailing_return - fee_round_trip - prior_28d_symbol_p90_full_spread_bps / 10000"
                if is_causal
                else "gross_trailing_return - fee_round_trip - pooled_asset_p90_full_spread_bps / 10000"
            ),
            "old_fixed_label_cost_replaced": True,
            "same_timestamp_spread_excluded": is_causal,
            "point_in_time": is_causal,
            "disclosure": (
                "causal prior-observation estimator"
                if is_causal
                else "static asset-level liquidity proxy; valid for target research but not an untouched-OOS cost claim"
            ),
        },
        "target": {side: asdict(spec) for side, spec in target_specs.items()},
        "rows": int(rows),
        # Coverage describes whether the selected spread estimator was available.
        # It must not imply a point-in-time guarantee for pooled/static research.
        "cost_observed_rows": int(observed_rows),
        "cost_observed_fraction": float(observed_rows / rows) if rows else 0.0,
        "fallback": "incumbent_target_soft_when_prior_spread_support_is_unavailable",
    }
