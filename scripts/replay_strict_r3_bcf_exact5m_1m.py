#!/usr/bin/env python3
"""Replay the frozen BCF/current-v5 dual gate with exact Kraken 1m execution.

This is a *fixed-score* validation producer.  It never refits the base,
residual, Geometry/K9, BCF, current-v5 or MC1 layers.  It changes one
execution assumption only: every eligible candidate enters on the Kraken
Futures one-minute open at ``decision + 5 minutes`` and is then replayed for
12 hours on complete one-minute bars with the frozen SimplePolicyOptimiser
geometry.  Candidates with a missing source minute or causal ATR are invalid
labels, not zero-return trades.

The output includes a reference replay using the historical source-aligned
policy labels, allowing the delayed-entry result to be compared under the same
dual-MC1 admission and portfolio auction contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.data_store import (  # noqa: E402
    PartitionedOHLCVStore,
    canonical_kraken_execution_1m_root,
)
from extreme_price_movements.portfolio_policy_replay import (  # noqa: E402
    normalise_candidate_table,
    replay_candidates,
)
from extreme_price_movements.simple_policy_optimiser import (  # noqa: E402
    simulate_and_score,
)
from scripts.report_strict_r3_mc1_d2_controlled_portfolio import (  # noqa: E402
    CAUSAL_AUCTION_CURVE,
    _metrics,
    _params,
)


HORIZON_MINUTES = 12 * 60
COST_BPS = 100.0
ATR_PERIODS = 14
MINUTE_ATR_WARMUP_HOURS = 100
IDENTITY = ("candidate_id", "__decision_ts__", "__symbol__")
POLICY_COLUMNS = (
    "policy_path_valid",
    "policy_gross_bps",
    "policy_net_bps",
    "policy_exit_bar_15m",
    "policy_entry_price",
    "policy_exit_price",
    "policy_exit_reason",
    "policy_label_available_ts",
    "policy_outcome_source",
    "policy_cost_bps",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_series(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="raise")


def _read_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {
        "candidate_id",
        "__decision_ts__",
        "__symbol__",
        "mc1_expected_bps",
        *POLICY_COLUMNS,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} lacks required columns: {missing}")
    frame["__decision_ts__"] = _utc_series(frame["__decision_ts__"])
    if frame["candidate_id"].duplicated().any():
        raise ValueError(f"{path} has duplicate candidate identities")
    return frame


def _read_request(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"timestamp", "symbol"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"candidate request lacks {missing}")
    frame = frame.rename(columns={"timestamp": "__decision_ts__", "symbol": "__symbol__"})
    frame["__decision_ts__"] = _utc_series(frame["__decision_ts__"])
    frame["__symbol__"] = frame["__symbol__"].astype(str)
    if frame.duplicated(list(IDENTITY[1:])).any():
        raise ValueError("candidate request has duplicate decision/symbol rows")
    return frame.loc[:, ["__decision_ts__", "__symbol__"]].copy()


def _policy(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    winner = payload.get("winner")
    if not isinstance(winner, dict):
        raise ValueError("policy JSON must have a winner object")
    keys = ("sl_mult", "trailing_activation_mult", "fixed_trailing_gap_mult")
    result = {key: float(winner[key]) for key in keys}
    if not all(np.isfinite(value) and value >= 0.0 for value in result.values()):
        raise ValueError("frozen policy geometry contains non-finite values")
    return result


def _hourly_atr(
    hourly_store: PartitionedOHLCVStore,
    symbol: str,
    *,
    end: pd.Timestamp,
) -> pd.Series:
    """Return Wilder-14 ATR indexed by the decision timestamp.

    A value at ``t`` uses the fully completed hourly bar ``[t-1h, t)`` and
    earlier bars only.  The calculation matches the causal alignment used by
    the frozen policy label materialiser.
    """
    frame = hourly_store.load(
        symbol,
        columns=["ts", "high", "low", "close"],
        end_ts=end - pd.Timedelta(hours=1),
    )
    if frame is None or frame.empty:
        return pd.Series(dtype=float)
    frame = frame.loc[:, ["high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce")
    frame = frame.loc[frame.index.notna()].sort_index()
    if frame.empty:
        return pd.Series(dtype=float)
    prior_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - prior_close).abs(),
            (frame["low"] - prior_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    complete = np.isfinite(frame.to_numpy(dtype=np.float64)).all(axis=1)
    true_range = true_range.where(complete)
    atr = true_range.ewm(alpha=1.0 / ATR_PERIODS, adjust=False, min_periods=ATR_PERIODS).mean()
    consecutive = pd.Series(complete, index=frame.index).rolling(
        ATR_PERIODS, min_periods=ATR_PERIODS
    ).sum().eq(ATR_PERIODS)
    atr = atr.where(consecutive)
    atr.index = atr.index + pd.Timedelta(hours=1)
    return atr


def _canonical_15m_aggregated_atr(
    symbol: str,
    *,
    decisions: pd.Series,
) -> pd.Series:
    """Return causal Wilder-14 ATR from the canonical 15-minute source.

    This is an explicitly source-aligned fallback for exact-one-minute
    historical replays when the legacy canonical hourly store has not yet
    been materialised.  Each value at ``t`` is constructed from the four
    complete 15-minute bars in ``[t-1h, t)`` and earlier bars only.  Loading
    is bounded at the last requested decision; later bars cannot influence
    the returned values.
    """
    decision_index = pd.DatetimeIndex(_utc_series(decisions)).sort_values()
    if decision_index.empty:
        return pd.Series(dtype=float)
    stem = symbol.lower().replace("/", "").replace("_", "")
    path = ROOT / "15m_ohlcv_perp" / f"{stem}_15m.parquet"
    if not path.exists():
        return pd.Series(dtype=float)
    frame = pd.read_parquet(path, columns=["open", "high", "low", "close"])
    frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce")
    frame = frame.loc[
        frame.index.notna()
        & ~frame.index.duplicated(keep="last")
        & (frame.index < decision_index.max())
    ].sort_index()
    if frame.empty:
        return pd.Series(dtype=float)
    ohlc = frame.loc[:, ["open", "high", "low", "close"]].apply(
        pd.to_numeric, errors="coerce"
    )
    finite = pd.Series(
        np.isfinite(ohlc.to_numpy(dtype=np.float64)).all(axis=1), index=ohlc.index
    )
    hourly = ohlc.resample("1h", label="left", closed="left").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    )
    complete = finite.resample("1h", label="left", closed="left").sum().eq(4)
    hourly.loc[~complete, :] = np.nan
    prior_close = hourly["close"].shift(1)
    true_range = pd.concat(
        [
            hourly["high"] - hourly["low"],
            (hourly["high"] - prior_close).abs(),
            (hourly["low"] - prior_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.ewm(alpha=1.0 / ATR_PERIODS, adjust=False, min_periods=ATR_PERIODS).mean()
    atr = atr.where(complete.rolling(ATR_PERIODS, min_periods=ATR_PERIODS).sum().eq(ATR_PERIODS))
    atr.index = atr.index + pd.Timedelta(hours=1)
    return atr


def _minute_aggregated_atr(
    minute_store: PartitionedOHLCVStore,
    symbol: str,
    *,
    decisions: pd.Series,
) -> pd.Series:
    """Causal Wilder-14 ATR from complete prior Kraken 1m bars.

    The source is aggregated only into fully completed hourly bars.  A
    100-hour complete warm-up both keeps the calculation strictly prior to the
    decision and makes the arbitrary local EWM initialisation negligible.
    Missing minutes invalidate the corresponding ATR rather than being filled.
    """
    decision_index = pd.DatetimeIndex(_utc_series(decisions)).sort_values()
    if decision_index.empty:
        return pd.Series(dtype=float)
    start = decision_index.min() - pd.Timedelta(hours=MINUTE_ATR_WARMUP_HOURS)
    end = decision_index.max()
    frame = minute_store.load(
        symbol,
        columns=["ts", "open", "high", "low", "close"],
        start_ts=start,
        end_ts=end,
    )
    if frame is None or frame.empty:
        return pd.Series(dtype=float)
    frame = frame.loc[:, ["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce")
    frame = frame.loc[frame.index.notna() & ~frame.index.duplicated(keep="last")].sort_index()
    if frame.empty:
        return pd.Series(dtype=float)
    hourly = frame.resample("1h", label="left", closed="left").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        minute_count=("close", "size"),
    )
    values = hourly.loc[:, ["open", "high", "low", "close"]].to_numpy(dtype=np.float64)
    complete = (
        hourly["minute_count"].eq(60)
        & np.isfinite(values).all(axis=1)
        & (values > 0.0).all(axis=1)
    )
    prior_close = hourly["close"].shift(1)
    true_range = pd.concat(
        [
            hourly["high"] - hourly["low"],
            (hourly["high"] - prior_close).abs(),
            (hourly["low"] - prior_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    true_range = true_range.where(complete)
    atr = true_range.ewm(alpha=1.0 / ATR_PERIODS, adjust=False, min_periods=MINUTE_ATR_WARMUP_HOURS).mean()
    stable = complete.rolling(MINUTE_ATR_WARMUP_HOURS, min_periods=MINUTE_ATR_WARMUP_HOURS).sum().eq(MINUTE_ATR_WARMUP_HOURS)
    atr = atr.where(stable)
    atr.index = atr.index + pd.Timedelta(hours=1)
    return atr


def _empty_labels(rows: pd.DataFrame, *, entry_delay_minutes: int) -> pd.DataFrame:
    out = rows.loc[:, ["candidate_id", "__decision_ts__", "__symbol__"]].copy()
    out["delayed_entry_ts"] = out["__decision_ts__"] + pd.Timedelta(minutes=int(entry_delay_minutes))
    out["policy_path_valid"] = False
    out["policy_gross_bps"] = np.nan
    out["policy_net_bps"] = np.nan
    out["policy_entry_price"] = np.nan
    out["policy_exit_price"] = np.nan
    out["policy_exit_minutes"] = np.nan
    out["policy_exit_timestamp"] = pd.NaT
    out["policy_exit_reason"] = "invalid_exact_1m_path"
    out["policy_atr"] = np.nan
    out["policy_atr_source"] = "unavailable"
    out["policy_label_available_ts"] = out["delayed_entry_ts"] + pd.Timedelta(hours=12)
    out["policy_outcome_source"] = "unavailable"
    out["policy_cost_bps"] = np.nan
    return out


def _one_minute_paths(
    minute: pd.DataFrame,
    entries: pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build candidate-local, complete 720-bar 1m paths without interpolation."""
    expected_columns = ("open", "high", "low", "close")
    if minute is None or minute.empty:
        empty = np.full((len(entries), HORIZON_MINUTES), np.nan, dtype=np.float32)
        return empty, empty.copy(), empty.copy(), empty.copy(), np.zeros(len(entries), dtype=bool)
    frame = minute.loc[:, list(expected_columns)].apply(pd.to_numeric, errors="coerce")
    frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce")
    frame = frame.loc[frame.index.notna() & ~frame.index.duplicated(keep="last")].sort_index()
    offsets = np.arange(HORIZON_MINUTES, dtype=np.int64) * int(pd.Timedelta(minutes=1).value)
    start_ns = pd.DatetimeIndex(entries).asi8.astype(np.int64, copy=False)
    target = (start_ns[:, None] + offsets[None, :]).reshape(-1)
    locations = frame.index.get_indexer(pd.DatetimeIndex(target, tz="UTC")).reshape(
        len(entries), HORIZON_MINUTES
    )
    present = locations >= 0
    safe = np.maximum(locations, 0)
    arrays: list[np.ndarray] = []
    for column in expected_columns:
        values = frame[column].to_numpy(dtype=np.float32, copy=False)
        selected = values[safe]
        selected[~present] = np.nan
        arrays.append(selected)
    finite = np.isfinite(np.stack(arrays, axis=0)).all(axis=(0, 2))
    positive = (arrays[0] > 0.0).all(axis=1) & (arrays[1] > 0.0).all(axis=1)
    valid = present.all(axis=1) & finite & positive
    return arrays[0], arrays[1], arrays[2], arrays[3], valid


def _labels_for_symbol(
    rows: pd.DataFrame,
    *,
    minute_store: PartitionedOHLCVStore,
    hourly_store: PartitionedOHLCVStore,
    policy: dict[str, float],
    atr_source: str,
    entry_delay_minutes: int,
) -> pd.DataFrame:
    out = _empty_labels(rows, entry_delay_minutes=entry_delay_minutes).reset_index(drop=True)
    symbol = str(rows["__symbol__"].iloc[0])
    decisions = _utc_series(rows["__decision_ts__"]).reset_index(drop=True)
    # The frozen policy-label contract binds ATR to the preceding hourly
    # signal timestamp.  The order is placed at the following decision time;
    # using the just-closed decision hour here would change the historical
    # policy geometry and leak a bar not available to the signal.
    signal_times = decisions - pd.Timedelta(hours=1)
    entries = decisions + pd.Timedelta(minutes=int(entry_delay_minutes))
    start = entries.min()
    end = entries.max() + pd.Timedelta(minutes=HORIZON_MINUTES)
    minute = minute_store.load(symbol, columns=["ts", "open", "high", "low", "close"], start_ts=start, end_ts=end)
    f_open, f_high, f_low, f_close, path_valid = _one_minute_paths(minute, entries)
    if atr_source == "minute_aggregated_100h":
        atr = _minute_aggregated_atr(minute_store, symbol, decisions=signal_times)
        atr_source_name = "kraken_1m_aggregated_wilder14_100h_warmup"
    elif atr_source == "canonical_hourly":
        atr = _hourly_atr(hourly_store, symbol, end=signal_times.max() + pd.Timedelta(hours=1))
        atr_source_name = "canonical_hourly_wilder14_prior_completed"
    elif atr_source == "canonical_15m_aggregated":
        atr = _canonical_15m_aggregated_atr(symbol, decisions=signal_times)
        atr_source_name = "canonical_15m_aggregated_wilder14_prior_completed"
    else:
        raise ValueError(f"unsupported atr source: {atr_source}")
    atr_values = signal_times.map(atr).to_numpy(dtype=np.float64)
    entry = f_open[:, 0].astype(np.float64)
    valid = path_valid & np.isfinite(atr_values) & (atr_values > 0.0) & np.isfinite(entry) & (entry > 0.0)
    if not valid.any():
        return out
    positions = np.flatnonzero(valid)
    run = pd.DataFrame(
        {
            "timestamp": entries.iloc[positions].to_numpy(),
            "symbol": np.repeat(symbol, len(positions)),
            "side": np.ones(len(positions), dtype=np.float32),
            "rank_pct": np.ones(len(positions), dtype=np.float32),
            "barrier_pct": atr_values[positions] / entry[positions],
            "expected_half_spread_bps": np.zeros(len(positions)),
            "exit_quote_half_spread_bps": np.zeros(len(positions)),
            "entry_slippage_proxy_bps": np.zeros(len(positions)),
            "market_mode": "perps",
        }
    )
    simulated = simulate_and_score(
        run,
        f_open[positions], f_high[positions], f_low[positions], f_close[positions],
        cost_pct=0.0,
        size_power=1.0,
        replay_timeframe="1m",
        market_mode="perps",
        sl_mult=policy["sl_mult"],
        sl_abs_cap_pct=0.0,
        trailing_activation_mult=policy["trailing_activation_mult"],
        trailing_activation_cap_pct=0.0,
        trailing_activation_max_bars=HORIZON_MINUTES,
        fixed_trailing_gap_mult=policy["fixed_trailing_gap_mult"],
        capital_protect_mfe_mult=0.0,
        adverse_exit_enabled=False,
        hard_tp_abs_pct=0.0,
        max_concurrent_trades=max(len(run), 1),
        max_concurrent_per_asset=max(len(run), 1),
        max_new_entries_per_bar=max(len(run), 1),
    )
    selected = np.asarray(simulated.get("selected_mask"), dtype=bool)
    if len(selected) != len(positions) or not selected.all():
        raise AssertionError("candidate-local exact replay unexpectedly applied a portfolio filter")
    gross = np.asarray(simulated["gross_returns"], dtype=np.float64) * 10_000.0
    exit_bars = np.asarray(simulated["exit_bars"], dtype=np.int64)
    exit_prices = np.asarray(simulated["exit_prices"], dtype=np.float64)
    exit_reason = np.asarray(simulated["exit_reason"], dtype=object)
    realised = np.isfinite(gross) & np.isfinite(exit_prices) & (exit_bars >= 0)
    loc = positions[realised]
    if len(loc):
        duration = exit_bars[realised] + 1
        out.loc[loc, "policy_path_valid"] = True
        out.loc[loc, "policy_gross_bps"] = gross[realised]
        out.loc[loc, "policy_net_bps"] = gross[realised] - COST_BPS
        out.loc[loc, "policy_entry_price"] = entry[loc]
        out.loc[loc, "policy_exit_price"] = exit_prices[realised]
        out.loc[loc, "policy_exit_minutes"] = duration
        out.loc[loc, "policy_exit_timestamp"] = entries.iloc[loc].to_numpy() + pd.to_timedelta(duration, unit="min")
        out.loc[loc, "policy_exit_reason"] = exit_reason[realised]
        out.loc[loc, "policy_atr"] = atr_values[loc]
        out.loc[loc, "policy_atr_source"] = atr_source_name
        out.loc[loc, "policy_label_available_ts"] = entries.iloc[loc].to_numpy() + pd.Timedelta(hours=12)
        out.loc[loc, "policy_outcome_source"] = "complete_kraken_1m_decision_plus_5"
        out.loc[loc, "policy_cost_bps"] = COST_BPS
    good = out["policy_path_valid"].to_numpy(bool)
    if good.any() and not np.allclose(
        out.loc[good, "policy_net_bps"].to_numpy(float),
        out.loc[good, "policy_gross_bps"].to_numpy(float) - COST_BPS,
        rtol=0.0,
        atol=1e-9,
    ):
        raise AssertionError("exact policy labels do not apply cost exactly once")
    return out


def _exact_labels(
    requested: pd.DataFrame,
    bcf: pd.DataFrame,
    *,
    data_root: Path,
    policy: dict[str, float],
    atr_source: str,
    entry_delay_minutes: int,
) -> pd.DataFrame:
    identities = bcf.loc[:, ["candidate_id", "__decision_ts__", "__symbol__"]].merge(
        requested,
        on=["__decision_ts__", "__symbol__"],
        how="inner",
        validate="one_to_one",
    )
    if len(identities) != len(requested):
        raise AssertionError("candidate request is not fully covered by BCF ledger")
    minute_store = PartitionedOHLCVStore(
        str(canonical_kraken_execution_1m_root(data_root)), timeframe="1m"
    )
    hourly_store = PartitionedOHLCVStore(
        # ``PartitionedOHLCVStore`` owns its ``ohlcv/`` child directory.  Pass
        # the exchange root, not that child, or every causal ATR lookup is
        # incorrectly empty (``.../ohlcv/ohlcv``).
        str(data_root / "exchanges" / "krakenfutures"), timeframe="1h"
    )
    parts: list[pd.DataFrame] = []
    total = identities["__symbol__"].nunique()
    for number, (_, group) in enumerate(identities.groupby("__symbol__", sort=True), start=1):
        parts.append(
            _labels_for_symbol(
                group.reset_index(drop=True),
                minute_store=minute_store,
                hourly_store=hourly_store,
                policy=policy,
                atr_source=atr_source,
                entry_delay_minutes=entry_delay_minutes,
            )
        )
        if number == 1 or number % 20 == 0 or number == total:
            print(json.dumps({"event": "exact_label_progress", "symbols_complete": number, "symbols_total": total}), flush=True)
    labels = pd.concat(parts, ignore_index=True).sort_values(
        ["__decision_ts__", "candidate_id"], kind="stable"
    ).reset_index(drop=True)
    if labels["candidate_id"].duplicated().any() or len(labels) != len(requested):
        raise AssertionError("exact label materialisation changed the requested identity set")
    return labels


def _original_labels(bcf: pd.DataFrame, requested: pd.DataFrame) -> pd.DataFrame:
    source = bcf.merge(requested, on=["__decision_ts__", "__symbol__"], how="inner", validate="one_to_one")
    out = source.loc[:, ["candidate_id", "__decision_ts__", "__symbol__", *POLICY_COLUMNS]].copy()
    valid = out["policy_path_valid"].fillna(False).astype(bool)
    exit_bar = pd.to_numeric(out["policy_exit_bar_15m"], errors="coerce")
    out["delayed_entry_ts"] = out["__decision_ts__"]
    out["policy_exit_minutes"] = (exit_bar + 1.0) * 15.0
    out["policy_exit_timestamp"] = out["__decision_ts__"] + pd.to_timedelta(out["policy_exit_minutes"], unit="min")
    out.loc[~valid, "policy_exit_timestamp"] = pd.NaT
    out["policy_atr"] = np.nan
    out["policy_atr_source"] = "historical_source_label"
    return out


def _portfolio_candidates(
    labels: pd.DataFrame,
    bcf: pd.DataFrame,
    current: pd.DataFrame,
    *,
    threshold_bps: float,
    entry_delay_minutes: int,
) -> pd.DataFrame:
    bcf_view = bcf.loc[:, ["candidate_id", "mc1_expected_bps"]].rename(columns={"mc1_expected_bps": "bcf_mc1_expected_bps"})
    current_view = current.loc[:, ["candidate_id", "mc1_expected_bps"]].rename(columns={"mc1_expected_bps": "current_mc1_expected_bps"})
    frame = labels.merge(bcf_view, on="candidate_id", how="left", validate="one_to_one").merge(
        current_view, on="candidate_id", how="left", validate="one_to_one"
    )
    valid = (
        frame["policy_path_valid"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(frame["policy_net_bps"], errors="coerce"))
        & np.isfinite(pd.to_numeric(frame["policy_gross_bps"], errors="coerce"))
        & np.isfinite(pd.to_numeric(frame["bcf_mc1_expected_bps"], errors="coerce"))
        & np.isfinite(pd.to_numeric(frame["current_mc1_expected_bps"], errors="coerce"))
    )
    admitted = frame.loc[
        valid
        & frame["bcf_mc1_expected_bps"].ge(float(threshold_bps))
        & frame["current_mc1_expected_bps"].ge(float(threshold_bps))
    ].copy()
    if admitted.empty:
        return pd.DataFrame()
    admitted["auction_rank"] = admitted.groupby("delayed_entry_ts", sort=False)[
        "bcf_mc1_expected_bps"
    ].rank(pct=True, method="average")
    candidate = pd.DataFrame(
        {
            "timestamp": _utc_series(admitted["delayed_entry_ts"]),
            "symbol": admitted["__symbol__"].astype(str),
            "side": "long",
            "strategy_id": f"strict_r3_bcf_current_v5_dual_mc1_exact{int(entry_delay_minutes)}m_long",
            "policy_archetype": f"strict_r3_bcf_current_v5_dual_mc1_exact{int(entry_delay_minutes)}m_long",
            "normalized_rank_score": admitted["auction_rank"].to_numpy(float),
            "strategy_rank_pct": admitted["auction_rank"].to_numpy(float),
            "base_strategy_threshold": 0.0,
            "calibrated_score": admitted["bcf_mc1_expected_bps"].to_numpy(float),
            "entry_price": pd.to_numeric(admitted["policy_entry_price"], errors="raise"),
            "exit_timestamp": _utc_series(admitted["policy_exit_timestamp"]),
            "exit_price": pd.to_numeric(admitted["policy_exit_price"], errors="raise"),
            "net_return": pd.to_numeric(admitted["policy_net_bps"], errors="raise") / 10_000.0,
            "gross_return": pd.to_numeric(admitted["policy_gross_bps"], errors="raise") / 10_000.0,
            "holding_bars": pd.to_numeric(admitted["policy_exit_minutes"], errors="raise"),
            "simple_policy_exit_reason": admitted["policy_exit_reason"].astype(str),
            "fees_bps": COST_BPS,
            "slippage_bps": 0.0,
            "expected_friction_bps": COST_BPS,
            "price_gap_bps": 0.0,
            "liquidity_capacity_weight": 1.0,
            "source_month": _utc_series(admitted["__decision_ts__"]).dt.strftime("%Y-%m"),
            "candidate_id": admitted["candidate_id"].astype(str),
            "mapped_expected_net_bps": admitted["bcf_mc1_expected_bps"].to_numpy(float),
            "bcf_mc1_expected_bps": admitted["bcf_mc1_expected_bps"].to_numpy(float),
            "current_mc1_expected_bps": admitted["current_mc1_expected_bps"].to_numpy(float),
            "original_decision_ts": _utc_series(admitted["__decision_ts__"]),
            "entry_delay_minutes": 5.0,
            "policy_outcome_available": True,
        }
    )
    return normalise_candidate_table(candidate)


def _run_portfolio(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    decisions, equity, _ = replay_candidates(
        candidates,
        _params(),
        mode="global_auction",
        ev_curve=CAUSAL_AUCTION_CURVE,
        market_mode="perps",
        initial_wallet=1000.0,
    )
    # ``normalise_candidate_table`` deliberately keeps the execution contract
    # lean, whereas the common portfolio reporter also audits outcome
    # availability.  Restore that evaluation-only provenance by the immutable
    # candidate index; it cannot alter auction ordering or realised returns.
    if "policy_outcome_available" not in decisions.columns:
        provenance = candidates.loc[:, ["policy_outcome_available"]].reset_index(drop=True)
        provenance.index.name = "candidate_index"
        decisions = decisions.merge(
            provenance,
            on="candidate_index",
            how="left",
            validate="many_to_one",
        )
        if decisions["policy_outcome_available"].isna().any():
            raise AssertionError("portfolio decision lacks exact-policy outcome provenance")
    metrics = _metrics(decisions, equity, "dual_bcf_priority", "2026")
    metrics["admitted_candidates_with_complete_outcomes"] = int(len(candidates))
    return decisions, equity, metrics


def _monthly(decisions: pd.DataFrame) -> pd.DataFrame:
    accepted = decisions.loc[decisions["accepted"].fillna(False).astype(bool)].copy()
    if accepted.empty:
        return pd.DataFrame(columns=["month", "trades", "net_ev_bps_per_trade", "net_sum_bps"])
    accepted["month"] = _utc_series(accepted["timestamp"]).dt.strftime("%Y-%m")
    return accepted.groupby("month", as_index=False).agg(
        trades=("accepted", "size"),
        net_ev_bps_per_trade=("position_net_return", lambda x: float(np.mean(x) * 10_000.0)),
        net_sum_bps=("position_net_return", lambda x: float(np.sum(x) * 10_000.0)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bcf-predictions", required=True, type=Path)
    parser.add_argument("--current-predictions", required=True, type=Path)
    parser.add_argument("--candidate-request", required=True, type=Path)
    parser.add_argument("--policy-json", required=True, type=Path)
    parser.add_argument("--data-root", default="data_perp", type=Path)
    parser.add_argument("--threshold-bps", type=float, default=30.0)
    parser.add_argument(
        "--entry-delay-minutes",
        type=int,
        default=5,
        help="fixed diagnostic entry delay from the completed decision candle",
    )
    parser.add_argument(
        "--atr-source",
        choices=("canonical_hourly", "minute_aggregated_100h"),
        default="canonical_hourly",
        help="causal ATR source used only for the frozen policy geometry",
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output already exists: {args.out_dir}")
    if int(args.entry_delay_minutes) < 0:
        raise ValueError("entry-delay-minutes must be non-negative")
    args.out_dir.mkdir(parents=True)
    bcf = _read_predictions(args.bcf_predictions)
    current = _read_predictions(args.current_predictions)
    request = _read_request(args.candidate_request)
    policy = _policy(args.policy_json)
    labels = _exact_labels(
        request,
        bcf,
        data_root=args.data_root,
        policy=policy,
        atr_source=str(args.atr_source),
        entry_delay_minutes=int(args.entry_delay_minutes),
    )
    labels.to_parquet(args.out_dir / "exact5m_policy_labels.parquet", index=False, compression="zstd")
    coverage = labels.assign(month=_utc_series(labels["__decision_ts__"]).dt.strftime("%Y-%m")).groupby("month", as_index=False).agg(
        requested_rows=("candidate_id", "size"),
        exact_valid_rows=("policy_path_valid", "sum"),
        exact_valid_fraction=("policy_path_valid", "mean"),
    )
    coverage.to_parquet(args.out_dir / "exact5m_coverage_by_month.parquet", index=False)
    results: list[dict[str, Any]] = []
    exact_arm = f"exact_1m_tplus{int(args.entry_delay_minutes)}"
    for name, label_frame in (("historical_source_reference", _original_labels(bcf, request)), (exact_arm, labels)):
        candidates = _portfolio_candidates(
            label_frame,
            bcf,
            current,
            threshold_bps=float(args.threshold_bps),
            entry_delay_minutes=int(args.entry_delay_minutes),
        )
        candidates.to_parquet(args.out_dir / f"{name}_candidates.parquet", index=False, compression="zstd")
        decisions, equity, metrics = _run_portfolio(candidates)
        decisions.to_parquet(args.out_dir / f"{name}_decisions.parquet", index=False, compression="zstd")
        equity.to_parquet(args.out_dir / f"{name}_equity.parquet", index=False, compression="zstd")
        month = _monthly(decisions)
        month["arm"] = name
        month.to_parquet(args.out_dir / f"{name}_monthly_metrics.parquet", index=False)
        metrics["arm"] = name
        results.append(metrics)
    metric_frame = pd.DataFrame(results)
    metric_frame.to_parquet(args.out_dir / "portfolio_metrics.parquet", index=False)
    manifest = {
        "schema": "strict_r3_bcf_current_v5_exact5m_1m_replay_v1",
        "purpose": "fixed-score, fixed-MC1 delayed-entry execution validation; not a retrain or promotion",
        "bcf_predictions": {"path": str(args.bcf_predictions), "sha256": _sha256(args.bcf_predictions)},
        "current_predictions": {"path": str(args.current_predictions), "sha256": _sha256(args.current_predictions)},
        "candidate_request": {"path": str(args.candidate_request), "sha256": _sha256(args.candidate_request)},
        "policy_json": {"path": str(args.policy_json), "sha256": _sha256(args.policy_json), "winner": policy},
        "entry": f"Kraken Futures 1m open at decision + {int(args.entry_delay_minutes)} minutes",
        "path": "720 complete 1m bars after entry; missing data invalidates the outcome",
        "atr": (
            "Wilder14 from complete prior Kraken 1m bars aggregated to hourly with "
            "a 100-hour causal warm-up"
            if args.atr_source == "minute_aggregated_100h"
            else "Wilder14 from canonical hourly source, aligned to prior completed hourly bars"
        ),
        "atr_source": str(args.atr_source),
        "cost_bps_once": COST_BPS,
        "admission": f"both frozen BCF/current MC1 expected EV >= {float(args.threshold_bps):g} bps",
        "auction_priority": "BCF MC1 expected EV",
        "portfolio": "existing long-only global auction, unchanged",
        "execution_store": str(canonical_kraken_execution_1m_root(args.data_root)),
        "requested_rows": int(len(request)),
        "exact_valid_rows": int(labels["policy_path_valid"].sum()),
        "exact_invalid_rows": int((~labels["policy_path_valid"]).sum()),
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "complete", "manifest": manifest, "metrics": results}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
