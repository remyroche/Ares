"""Vectorised 15-minute labels for the schema-v2 frozen execution policy."""

from __future__ import annotations

import numba as nb
import numpy as np
import pandas as pd


HORIZON_BARS = 48
COST_BPS = 100.0


@nb.njit(cache=True)
def _replay_coarse_policy(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    starts: np.ndarray,
    entry: np.ndarray,
    atr: np.ndarray,
    side_sign: np.ndarray,
    horizon_bars: int,
    stop_loss_atr: float,
    trailing_activation_atr: float,
    trailing_giveback_atr: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Replay one fixed policy on a coarse OHLC path.

    The helper is intentionally frequency agnostic. It preserves the
    canonical pessimistic stop precedence and does not let a bar both arm and
    trigger its own trailing exit. This makes a one-hour fallback conservative
    relative to an unavailable 15-minute path without pretending that the
    coarse path has intrabar ordering precision.
    """
    rows = len(starts)
    gross = np.full(rows, np.nan, np.float64)
    exit_bar = np.full(rows, -1, np.int16)
    reason = np.full(rows, -1, np.int8)
    exit_price = np.full(rows, np.nan, np.float64)
    for row in range(rows):
        start = starts[row]
        price = entry[row]
        risk = atr[row]
        sign = side_sign[row]
        if (
            start < 0 or start + horizon_bars > len(high)
            or not np.isfinite(price) or not np.isfinite(risk)
            or price <= 0.0 or risk <= 0.0
        ):
            continue
        maximum_favourable = 0.0
        armed = False
        resolved = False
        for bar in range(horizon_bars):
            position = start + bar
            bar_high = high[position]
            bar_low = low[position]
            bar_close = close[position]
            if not (
                np.isfinite(bar_high)
                and np.isfinite(bar_low)
                and np.isfinite(bar_close)
            ):
                resolved = True
                break
            stop_price = price - sign * stop_loss_atr * risk
            stop_hit = bar_low <= stop_price if sign > 0 else bar_high >= stop_price
            if stop_hit:
                exit_price[row] = stop_price
                gross[row] = -stop_loss_atr * risk / price * 10_000.0
                exit_bar[row] = bar
                reason[row] = 0
                resolved = True
                break
            if maximum_favourable > trailing_activation_atr * risk:
                armed = True
            if armed:
                locked = max(
                    maximum_favourable - trailing_giveback_atr * risk,
                    0.0,
                )
                trail_price = price + sign * locked
                trail_hit = bar_low <= trail_price if sign > 0 else bar_high >= trail_price
                if trail_hit:
                    exit_price[row] = trail_price
                    gross[row] = locked / price * 10_000.0
                    exit_bar[row] = bar
                    reason[row] = 1
                    resolved = True
                    break
            favourable = bar_high - price if sign > 0 else price - bar_low
            if favourable > maximum_favourable:
                maximum_favourable = favourable
        if not resolved:
            terminal = close[start + horizon_bars - 1]
            if np.isfinite(terminal):
                exit_price[row] = terminal
                gross[row] = sign * (terminal - price) / price * 10_000.0
                exit_bar[row] = horizon_bars - 1
                reason[row] = 2
    return gross, exit_bar, reason, exit_price


def replay_frozen_policy_15m(
    candidates: pd.DataFrame,
    bars: pd.DataFrame,
    *,
    stop_loss_atr: float = 3.0,
    trailing_activation_atr: float = 0.5,
    trailing_giveback_atr: float = 0.25,
    cost_bps: float = COST_BPS,
) -> pd.DataFrame:
    """Replay one symbol's candidates while retaining every invalid identity."""
    required = {"candidate_id", "__decision_ts__", "side_name", "atr_1h"}
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"policy candidates lack {missing}")
    if candidates["candidate_id"].duplicated().any():
        raise ValueError("policy candidates have duplicate identities")
    market = bars.copy()
    market.index = pd.to_datetime(market.index, utc=True, errors="raise")
    market = market.sort_index()
    if market.index.duplicated().any():
        raise ValueError("15-minute source has duplicate timestamps")
    if len(market):
        full_index = pd.date_range(market.index.min(), market.index.max(), freq="15min", tz="UTC")
        market = market.reindex(full_index)
    decision = pd.to_datetime(candidates["__decision_ts__"], utc=True, errors="raise")
    starts = market.index.get_indexer(pd.DatetimeIndex(decision))
    entry = np.full(len(candidates), np.nan, dtype=float)
    valid_start = starts >= 0
    entry[valid_start] = pd.to_numeric(market["open"], errors="coerce").to_numpy(float)[starts[valid_start]]
    atr = pd.to_numeric(candidates["atr_1h"], errors="coerce").to_numpy(float)
    side = candidates["side_name"].astype(str).str.lower()
    if not side.isin(["long", "short"]).all():
        raise ValueError("policy candidates contain a noncanonical side")
    sign = np.where(side.eq("long"), 1, -1).astype(np.int8)
    for name, value in (
        ("stop_loss_atr", stop_loss_atr),
        ("trailing_activation_atr", trailing_activation_atr),
        ("trailing_giveback_atr", trailing_giveback_atr),
        ("cost_bps", cost_bps),
    ):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    gross, exit_bar, reason, exit_price = _replay_coarse_policy(
        pd.to_numeric(market["high"], errors="coerce").to_numpy(float),
        pd.to_numeric(market["low"], errors="coerce").to_numpy(float),
        pd.to_numeric(market["close"], errors="coerce").to_numpy(float),
        starts.astype(np.int64), entry, atr, sign,
        HORIZON_BARS, float(stop_loss_atr),
        float(trailing_activation_atr), float(trailing_giveback_atr),
    )
    # A regularised cache can contain synthetic zero-volume flat candles when
    # its exchange tail was not refreshed.  That is missing supervision, not
    # an economic timeout at exactly minus the declared cost.  Genuine
    # no-trade intervals inside an otherwise moving H12 path remain valid.
    if "volume" in market.columns:
        volume = pd.to_numeric(market["volume"], errors="coerce").to_numpy(float)
        high = pd.to_numeric(market["high"], errors="coerce").to_numpy(float)
        low = pd.to_numeric(market["low"], errors="coerce").to_numpy(float)
        synthetic_flat = np.zeros(len(candidates), dtype=bool)
        for row, start in enumerate(starts):
            if start < 0 or start + HORIZON_BARS > len(market):
                continue
            path_high = high[start:start + HORIZON_BARS]
            path_low = low[start:start + HORIZON_BARS]
            path_volume = volume[start:start + HORIZON_BARS]
            synthetic_flat[row] = bool(
                np.isfinite(path_high).all()
                and np.isfinite(path_low).all()
                and np.isfinite(path_volume).all()
                and np.nanmax(path_high) == np.nanmin(path_low)
                and np.all(path_volume <= 0.0)
            )
        gross[synthetic_flat] = np.nan
        exit_bar[synthetic_flat] = -1
        reason[synthetic_flat] = -1
        exit_price[synthetic_flat] = np.nan
    output = candidates.copy()
    output["policy_entry_price"] = entry
    output["policy_exit_price"] = exit_price
    output["policy_gross_bps"] = gross
    output["policy_net_bps"] = gross - float(cost_bps)
    output["policy_exit_bar_15m"] = exit_bar
    output["policy_exit_reason"] = pd.Categorical.from_codes(
        np.where(reason < 0, 3, reason),
        categories=("stop_loss", "trailing", "timeout_h12", "invalid_path"),
    ).astype(str)
    output["policy_path_valid"] = np.isfinite(gross)
    output["policy_label_available_ts"] = decision + pd.Timedelta(hours=12)
    output["policy_cost_bps"] = np.where(
        output["policy_path_valid"], float(cost_bps), np.nan,
    )
    if not np.allclose(
        output.loc[output["policy_path_valid"], "policy_net_bps"],
        output.loc[output["policy_path_valid"], "policy_gross_bps"] - float(cost_bps),
        rtol=0.0, atol=1e-12,
    ):
        raise AssertionError("policy cost was not applied exactly once")
    return output


def causal_hourly_atr_from_hourly(bars: pd.DataFrame) -> pd.Series:
    """Return Wilder ATR(14) indexed by its decision-time availability.

    An hourly candle stamped t covers [t, t+1h). Its true range is therefore
    available at t+1h. Mapping this series to a candidate decision timestamp
    cannot inspect its execution bar or later bars.
    """
    market = bars.loc[:, ["open", "high", "low", "close"]].apply(
        pd.to_numeric, errors="coerce",
    )
    market.index = pd.to_datetime(market.index, utc=True, errors="raise")
    market = market.sort_index()
    if market.index.duplicated().any():
        raise ValueError("hourly proxy source has duplicate timestamps")
    if len(market):
        market = market.reindex(
            pd.date_range(market.index.min(), market.index.max(), freq="1h", tz="UTC")
        )
    complete = market.notna().all(axis=1)
    previous = market["close"].shift(1)
    true_range = pd.concat(
        [
            market["high"] - market["low"],
            (market["high"] - previous).abs(),
            (market["low"] - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.ewm(alpha=1.0 / 14.0, adjust=False, min_periods=14).mean()
    atr = atr.where(complete.rolling(14, min_periods=14).sum().eq(14))
    atr.index = atr.index + pd.Timedelta(hours=1)
    return atr


def replay_policy_hourly_proxy(
    candidates: pd.DataFrame,
    bars: pd.DataFrame,
    *,
    stop_loss_atr: float,
    trailing_activation_atr: float,
    trailing_giveback_atr: float,
    timeout_hours: int = 12,
    cost_bps: float = COST_BPS,
) -> pd.DataFrame:
    """Replay the policy on hourly OHLC only when finer paths are unavailable.

    The returned rows are explicitly labelled as one-hour proxies. Existing
    exact/15-minute outcomes must retain precedence in any merged label ledger.
    """
    required = {
        "candidate_id", "__decision_ts__", "side_name", "atr_1h",
    }
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"hourly policy candidates lack {missing}")
    if candidates["candidate_id"].duplicated().any():
        raise ValueError("hourly policy candidates have duplicate identities")
    if timeout_hours < 1:
        raise ValueError("timeout_hours must be positive")
    for name, value in (
        ("stop_loss_atr", stop_loss_atr),
        ("trailing_activation_atr", trailing_activation_atr),
        ("trailing_giveback_atr", trailing_giveback_atr),
        ("cost_bps", cost_bps),
    ):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")

    market = bars.loc[:, ["open", "high", "low", "close"]].apply(
        pd.to_numeric, errors="coerce",
    )
    market.index = pd.to_datetime(market.index, utc=True, errors="raise")
    market = market.sort_index()
    if market.index.duplicated().any():
        raise ValueError("hourly proxy source has duplicate timestamps")
    if len(market):
        market = market.reindex(
            pd.date_range(market.index.min(), market.index.max(), freq="1h", tz="UTC")
        )
    decision = pd.to_datetime(candidates["__decision_ts__"], utc=True, errors="raise")
    starts = market.index.get_indexer(pd.DatetimeIndex(decision))
    entry = np.full(len(candidates), np.nan, dtype=float)
    valid_start = starts >= 0
    if valid_start.any():
        entry[valid_start] = market["open"].to_numpy(float)[starts[valid_start]]
    atr = pd.to_numeric(candidates["atr_1h"], errors="coerce").to_numpy(float)
    fallback_atr = causal_hourly_atr_from_hourly(market)
    missing_atr = ~np.isfinite(atr) | (atr <= 0.0)
    if missing_atr.any():
        atr[missing_atr] = decision[missing_atr].map(fallback_atr).to_numpy(float)
    side = candidates["side_name"].astype(str).str.lower()
    if not side.isin(["long", "short"]).all():
        raise ValueError("hourly policy candidates contain a noncanonical side")
    sign = np.where(side.eq("long"), 1, -1).astype(np.int8)
    gross, exit_bar, reason, exit_price = _replay_coarse_policy(
        market["high"].to_numpy(float),
        market["low"].to_numpy(float),
        market["close"].to_numpy(float),
        starts.astype(np.int64),
        entry,
        atr,
        sign,
        int(timeout_hours),
        float(stop_loss_atr),
        float(trailing_activation_atr),
        float(trailing_giveback_atr),
    )
    output = candidates.copy()
    output["policy_atr"] = atr
    output["policy_atr_source"] = np.where(
        missing_atr & np.isfinite(atr),
        "hourly_proxy_wilder14",
        "provided_causal_atr",
    )
    output["policy_entry_price"] = entry
    output["policy_exit_price"] = exit_price
    output["policy_gross_bps"] = gross
    output["policy_net_bps"] = gross - float(cost_bps)
    output["policy_exit_bar_1h"] = exit_bar
    # Downstream portfolio code consumes a 15-minute bar ordinal. A proxy
    # exit is placed at the end of its hourly candle, never earlier.
    output["policy_exit_bar_15m"] = np.where(
        exit_bar >= 0, (exit_bar.astype(np.int32) + 1) * 4 - 1, -1,
    ).astype(np.int16)
    output["policy_exit_reason"] = pd.Categorical.from_codes(
        np.where(reason < 0, 3, reason),
        categories=(
            "hourly_proxy_sl", "hourly_proxy_trailing",
            "hourly_proxy_timeout", "invalid_path",
        ),
    ).astype(str)
    output["policy_path_valid"] = np.isfinite(gross)
    output["policy_label_available_ts"] = decision + pd.Timedelta(hours=timeout_hours)
    output["policy_cost_bps"] = np.where(
        output["policy_path_valid"], float(cost_bps), np.nan,
    )
    output["policy_outcome_source"] = np.where(
        output["policy_path_valid"], "hourly_ohlc_proxy", "unavailable",
    )
    output["policy_market_data_source"] = output["policy_outcome_source"]
    output["policy_market_data_quality"] = np.where(
        output["policy_path_valid"],
        "complete_12x1h_conservative_ordering",
        "incomplete_hourly_path",
    )
    valid = output["policy_path_valid"].to_numpy(bool)
    if valid.any() and not np.allclose(
        output.loc[valid, "policy_net_bps"],
        output.loc[valid, "policy_gross_bps"] - float(cost_bps),
        rtol=0.0,
        atol=1e-12,
    ):
        raise AssertionError("hourly proxy cost was not applied exactly once")
    return output
