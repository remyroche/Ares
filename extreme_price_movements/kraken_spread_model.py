from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from extreme_price_movements.data_store import (
    _ARCHIVE_USER_AGENT,
    _fetch_kraken_futures_charts_ohlcv,
    _public_data_session,
    make_perp_exchange,
)
from extreme_price_movements.utils import tprint


KRAKEN_FUTURES_TICKERS_URL = "https://futures.kraken.com/derivatives/api/v3/tickers"
DEFAULT_OUTPUT_DIR = Path("data_perp/exchanges/krakenfutures/spread_model")
DEFAULT_SNAPSHOT_OUTPUT_DIR = Path("data_perp/exchanges/krakenfutures/spread_snapshots")
SPREAD_CANDLE_FEATURES = [
    "hl_range_bps",
    "abs_return_bps",
    "body_bps",
    "upper_wick_bps",
    "lower_wick_bps",
    "wick_to_range",
    "close_location",
    "gap_bps",
    "log_candle_volume",
    "log_candle_quote_volume",
]
SPREAD_CROSS_SECTIONAL_FEATURES = [
    "candle_volume_rank",
    "candle_quote_volume_rank",
    "hl_range_bps_rank",
    "abs_return_bps_rank",
    "body_bps_rank",
    "wick_to_range_rank",
    "gap_bps_rank",
]
SPREAD_TIME_FEATURES = [
    "minute_of_day_sin",
    "minute_of_day_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "is_weekend",
]
SPREAD_OHLCV_ROLLING_FEATURES = [
    "asset_hl_range_bps_roll_mean_3",
    "asset_abs_return_bps_roll_mean_3",
    "asset_log_candle_quote_volume_roll_mean_3",
    "asset_log_candle_quote_volume_lag1",
]
SPREAD_MODEL_FEATURES = [
    *SPREAD_CANDLE_FEATURES,
    *SPREAD_CROSS_SECTIONAL_FEATURES,
    *SPREAD_TIME_FEATURES,
    *SPREAD_OHLCV_ROLLING_FEATURES,
]
EPS = 1e-12


@dataclass(frozen=True)
class SpreadUniverseItem:
    base: str
    perp_symbol: str
    perp_market_id: str
    tick_size: float


def _normalise_datetime_index(index: Any, *, floor: str = "min") -> pd.DatetimeIndex:
    idx = pd.to_datetime(index, utc=True, errors="coerce")
    idx = pd.DatetimeIndex(idx)
    if floor:
        idx = idx.floor(floor)
    return idx


def _market_tradeable(market: Dict[str, Any]) -> bool:
    info = market.get("info") if isinstance(market.get("info"), dict) else {}
    if market.get("active") is False:
        return False
    status = str(info.get("status") or info.get("marketStatus") or "").lower()
    if status and status not in {"online", "open", "trading", "enabled"}:
        return False
    for key in ("tradeable", "tradable", "active", "isTrading"):
        if key in info and str(info.get(key)).lower() in {"0", "false", "no", "disabled"}:
            return False
    return True


def _tick_size_from_market(market: Dict[str, Any]) -> float:
    info = market.get("info") if isinstance(market.get("info"), dict) else {}
    for key in (
        "tickSize",
        "tick_size",
        "priceTickSize",
        "price_tick_size",
        "tick_size_price",
    ):
        value = market.get(key, info.get(key))
        if value is None:
            continue
        try:
            tick = float(value)
        except Exception:
            continue
        if np.isfinite(tick) and tick > 0.0:
            return float(tick)
    precision = market.get("precision") if isinstance(market.get("precision"), dict) else {}
    price_precision = precision.get("price")
    try:
        value = float(price_precision)
    except Exception:
        value = float("nan")
    if np.isfinite(value) and value > 0.0:
        if value < 1.0:
            return float(value)
        if float(value).is_integer() and value <= 12:
            return float(10.0 ** (-int(value)))
    return 0.01


def resolve_spread_universe(
    perp_exchange: Any,
    *,
    symbols: Optional[Iterable[str]] = None,
) -> Tuple[List[SpreadUniverseItem], List[Dict[str, Any]]]:
    allowed = {str(s).upper().strip() for s in symbols or [] if str(s).strip()}
    out: List[SpreadUniverseItem] = []
    audit: List[Dict[str, Any]] = []
    for market in (getattr(perp_exchange, "markets", {}) or {}).values():
        if not isinstance(market, dict) or not _market_tradeable(market):
            continue
        if not bool(market.get("swap")):
            continue
        quote = str(market.get("quote") or "").upper()
        settle = str(market.get("settle") or "").upper()
        if quote != "USD" and settle != "USD":
            continue
        base = str(market.get("base") or "").upper()
        perp_symbol = str(market.get("symbol") or "")
        if not base or not perp_symbol:
            continue
        market_id = str(market.get("id") or (market.get("info") or {}).get("symbol") or "")
        if (
            allowed
            and base not in allowed
            and perp_symbol.upper() not in allowed
            and market_id.upper() not in allowed
        ):
            continue
        item = SpreadUniverseItem(
            base=base,
            perp_symbol=perp_symbol,
            perp_market_id=market_id or perp_symbol,
            tick_size=_tick_size_from_market(market),
        )
        out.append(item)
        audit.append({**asdict(item), "status": "eligible"})
    out.sort(key=lambda x: x.perp_symbol)
    return out, audit


def _item_lookup(universe: Sequence[SpreadUniverseItem]) -> Dict[str, SpreadUniverseItem]:
    out: Dict[str, SpreadUniverseItem] = {}
    for item in universe:
        out[item.perp_symbol.upper()] = item
        out[item.perp_market_id.upper()] = item
        out[item.base.upper()] = item
    return out


def parse_kraken_futures_tickers_payload(
    payload: Dict[str, Any],
    *,
    universe: Sequence[SpreadUniverseItem],
    snapshot_ts: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    if not isinstance(payload, dict):
        return pd.DataFrame()
    if str(payload.get("result", "success")).lower() not in {"success", ""}:
        raise RuntimeError(f"Kraken Futures tickers error: {payload}")
    tickers = payload.get("tickers")
    if not isinstance(tickers, list) or not tickers:
        return pd.DataFrame()
    if snapshot_ts is None:
        snapshot_ts = pd.to_datetime(
            payload.get("serverTime") or pd.Timestamp.now(tz="UTC"),
            utc=True,
            errors="coerce",
        )
    snapshot_ts = pd.Timestamp(snapshot_ts)
    if snapshot_ts.tzinfo is None:
        snapshot_ts = snapshot_ts.tz_localize("UTC")
    else:
        snapshot_ts = snapshot_ts.tz_convert("UTC")
    minute_ts = snapshot_ts.floor("min")
    lookup = _item_lookup(universe)
    rows: List[Dict[str, Any]] = []
    for ticker in tickers:
        if not isinstance(ticker, dict):
            continue
        market_id = str(ticker.get("symbol") or "")
        item = lookup.get(market_id.upper())
        if item is None:
            pair = str(ticker.get("pair") or "").replace(":", "/")
            item = lookup.get(pair.upper())
        if item is None:
            continue
        try:
            bid = float(ticker.get("bid"))
            ask = float(ticker.get("ask"))
        except Exception:
            continue
        if not np.isfinite(bid) or not np.isfinite(ask) or bid <= 0.0 or ask <= bid:
            continue
        mid = (bid + ask) * 0.5
        tick = max(float(item.tick_size), EPS)
        rows.append(
            {
                "timestamp": minute_ts,
                "observed_ts": snapshot_ts,
                "symbol": item.perp_symbol,
                "perp_market_id": item.perp_market_id,
                "base": item.base,
                "bid": bid,
                "ask": ask,
                "bid_size": pd.to_numeric(ticker.get("bidSize"), errors="coerce"),
                "ask_size": pd.to_numeric(ticker.get("askSize"), errors="coerce"),
                "mid": mid,
                "spread_bps": 10000.0 * (ask - bid) / (abs(mid) + EPS),
                "spread_ticks": (ask - bid) / tick,
                "min_tick_spread_bps": 10000.0 * tick / (abs(mid) + EPS),
                "tick_size": tick,
            }
        )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).set_index("timestamp").sort_index()
    return out.replace([np.inf, -np.inf], np.nan)


def fetch_kraken_futures_ticker_spreads(
    *,
    universe: Sequence[SpreadUniverseItem],
    session: Any = None,
    timeout: int = 30,
) -> pd.DataFrame:
    session = session or _public_data_session()
    response = session.get(
        KRAKEN_FUTURES_TICKERS_URL,
        timeout=timeout,
        headers={"User-Agent": _ARCHIVE_USER_AGENT},
    )
    response.raise_for_status()
    return parse_kraken_futures_tickers_payload(
        response.json(),
        universe=universe,
    )


def collect_kraken_futures_ticker_spreads(
    *,
    universe: Sequence[SpreadUniverseItem],
    snapshot_count: int = 1,
    snapshot_interval_seconds: float = 0.0,
    session: Any = None,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    count = max(1, int(snapshot_count))
    session = session or _public_data_session()
    for i in range(count):
        frame = fetch_kraken_futures_ticker_spreads(
            universe=universe,
            session=session,
        )
        if not frame.empty:
            frames.append(frame)
        if i + 1 < count and snapshot_interval_seconds > 0.0:
            time.sleep(float(snapshot_interval_seconds))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=0).sort_index()
    out = out.reset_index().drop_duplicates(["timestamp", "symbol"], keep="last")
    return out.set_index("timestamp").sort_index()


def _seconds_until_next_hour(
    now: Optional[pd.Timestamp] = None,
    *,
    grace_seconds: float = 5.0,
) -> float:
    ts = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    hour_start = ts.floor("h")
    if (ts - hour_start).total_seconds() <= max(float(grace_seconds), 0.0):
        return 0.0
    next_hour = hour_start + pd.Timedelta(hours=1)
    return max(float((next_hour - ts).total_seconds()), 0.0)


def _spread_snapshot_summary(
    frame: pd.DataFrame,
    *,
    universe_count: int,
    endpoint: str = KRAKEN_FUTURES_TICKERS_URL,
) -> Dict[str, Any]:
    if frame is None or frame.empty:
        return {
            "rows": 0,
            "symbols": 0,
            "universe_count": int(universe_count),
            "source_endpoint": endpoint,
        }
    idx = _normalise_datetime_index(frame.index, floor="min")
    spread_bps = pd.to_numeric(frame.get("spread_bps"), errors="coerce")
    spread_ticks = pd.to_numeric(frame.get("spread_ticks"), errors="coerce")
    per_asset = (
        frame.assign(
            spread_bps=spread_bps.to_numpy(dtype=np.float64),
            spread_ticks=spread_ticks.to_numpy(dtype=np.float64),
        )
        .groupby("symbol", sort=True)
        .agg(
            rows=("spread_bps", "size"),
            average_spread_bps=("spread_bps", "mean"),
            median_spread_bps=("spread_bps", "median"),
            p75_spread_bps=("spread_bps", lambda x: float(np.nanpercentile(x, 75))),
            average_spread_ticks=("spread_ticks", "mean"),
        )
        .reset_index()
    )
    return {
        "rows": int(len(frame)),
        "symbols": int(frame["symbol"].nunique()) if "symbol" in frame.columns else 0,
        "universe_count": int(universe_count),
        "timestamp_min": idx.min().isoformat() if len(idx) else None,
        "timestamp_max": idx.max().isoformat() if len(idx) else None,
        "source_endpoint": endpoint,
        "spread_bps_mean": float(spread_bps.mean()),
        "spread_bps_median": float(spread_bps.median()),
        "spread_bps_p75": float(np.nanpercentile(spread_bps, 75)),
        "spread_ticks_mean": float(spread_ticks.mean()),
        "per_asset_average_spread": per_asset.to_dict(orient="records"),
    }


def _append_history_parquet(
    history_path: Path,
    frame: pd.DataFrame,
    *,
    dedupe_cols: Sequence[str] = ("symbol",),
) -> Tuple[Path, int]:
    new_frame = frame.copy() if frame is not None else pd.DataFrame()
    if new_frame.empty:
        return history_path, 0
    new_frame.index = _normalise_datetime_index(new_frame.index, floor="min")
    new_frame.index.name = "timestamp"
    if history_path.exists():
        old_frame = pd.read_parquet(history_path)
        old_frame.index = _normalise_datetime_index(old_frame.index, floor="min")
        old_frame.index.name = "timestamp"
        combined = pd.concat([old_frame, new_frame], axis=0, sort=False)
    else:
        combined = new_frame
    if dedupe_cols:
        tmp = combined.reset_index()
        subset = ["timestamp", *[col for col in dedupe_cols if col in tmp.columns]]
        if len(subset) > 1:
            tmp = tmp.drop_duplicates(subset=subset, keep="last")
        combined = tmp.set_index("timestamp")
    combined = combined.sort_index()
    combined.to_parquet(history_path, compression="zstd")
    return history_path, int(len(combined))


def save_spread_snapshot_collection(
    frame: pd.DataFrame,
    *,
    universe_audit: Sequence[Dict[str, Any]],
    candles: Optional[pd.DataFrame] = None,
    training: Optional[pd.DataFrame] = None,
    candle_audit: Optional[Sequence[Dict[str, Any]]] = None,
    output_dir: Path | str = DEFAULT_SNAPSHOT_OUTPUT_DIR,
    run_id: Optional[str] = None,
) -> Tuple[Path, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_id or pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
    frame_to_write = frame.copy() if frame is not None else pd.DataFrame()
    if not frame_to_write.empty:
        frame_to_write.index = _normalise_datetime_index(frame_to_write.index, floor="min")
        frame_to_write.index.name = "timestamp"
    parquet_path = out_dir / f"kraken_futures_perp_spreads_{run_id}.parquet"
    latest_path = out_dir / "latest.parquet"
    history_path = out_dir / "history.parquet"
    candles_path = out_dir / f"kraken_futures_perp_1m_candles_{run_id}.parquet"
    latest_candles_path = out_dir / "latest_candles.parquet"
    history_candles_path = out_dir / "history_candles.parquet"
    training_path = out_dir / f"kraken_futures_perp_spread_training_{run_id}.parquet"
    latest_training_path = out_dir / "latest_training.parquet"
    history_training_path = out_dir / "history_training.parquet"
    summary_path = out_dir / f"kraken_futures_perp_spreads_{run_id}_summary.json"
    latest_summary_path = out_dir / "latest_summary.json"
    frame_to_write.to_parquet(parquet_path, compression="zstd")
    frame_to_write.to_parquet(latest_path, compression="zstd")
    _, history_rows = _append_history_parquet(history_path, frame_to_write)
    candles_to_write = candles.copy() if candles is not None else pd.DataFrame()
    history_candle_rows = 0
    if not candles_to_write.empty:
        candles_to_write.index = _normalise_datetime_index(candles_to_write.index, floor="min")
        candles_to_write.index.name = "timestamp"
        candles_to_write.to_parquet(candles_path, compression="zstd")
        candles_to_write.to_parquet(latest_candles_path, compression="zstd")
        _, history_candle_rows = _append_history_parquet(history_candles_path, candles_to_write)
    training_to_write = training.copy() if training is not None else pd.DataFrame()
    history_training_rows = 0
    if not training_to_write.empty:
        training_to_write.index = _normalise_datetime_index(training_to_write.index, floor="min")
        training_to_write.index.name = "timestamp"
        training_to_write.to_parquet(training_path, compression="zstd")
        training_to_write.to_parquet(latest_training_path, compression="zstd")
        _, history_training_rows = _append_history_parquet(history_training_path, training_to_write)
    summary = _spread_snapshot_summary(
        frame_to_write,
        universe_count=len(universe_audit),
    )
    summary.update(
        {
            "run_id": run_id,
            "created_ts": pd.Timestamp.now(tz="UTC").isoformat(),
            "snapshot_path": str(parquet_path.resolve()),
            "latest_snapshot_path": str(latest_path.resolve()),
            "history_snapshot_path": str(history_path.resolve()),
            "candles_path": str(candles_path.resolve()) if not candles_to_write.empty else None,
            "latest_candles_path": str(latest_candles_path.resolve()) if not candles_to_write.empty else None,
            "history_candles_path": str(history_candles_path.resolve()) if history_candle_rows else None,
            "training_path": str(training_path.resolve()) if not training_to_write.empty else None,
            "latest_training_path": str(latest_training_path.resolve()) if not training_to_write.empty else None,
            "history_training_path": str(history_training_path.resolve()) if history_training_rows else None,
            "candle_rows": int(len(candles_to_write)),
            "training_rows": int(len(training_to_write)),
            "history_rows": int(history_rows),
            "history_candle_rows": int(history_candle_rows),
            "history_training_rows": int(history_training_rows),
            "candle_audit": list(candle_audit or []),
            "universe_audit": list(universe_audit),
        }
    )
    payload = json.dumps(_jsonify(summary), indent=2, sort_keys=True, allow_nan=False) + "\n"
    summary_path.write_text(payload, encoding="utf-8")
    latest_summary_path.write_text(payload, encoding="utf-8")
    return parquet_path, summary_path


def fetch_kraken_futures_1m_candles_for_spread_minutes(
    perp_exchange: Any,
    symbol: str,
    minutes: Sequence[pd.Timestamp],
    *,
    pad_minutes: int = 1,
) -> pd.DataFrame:
    idx = _normalise_datetime_index(minutes, floor="min")
    idx = idx[~idx.isna()].sort_values().unique()
    if len(idx) == 0:
        return pd.DataFrame()
    start = pd.Timestamp(idx.min()) - pd.Timedelta(minutes=max(int(pad_minutes), 0))
    end = pd.Timestamp(idx.max()) + pd.Timedelta(minutes=max(int(pad_minutes), 0) + 1)
    frame = _fetch_kraken_futures_charts_ohlcv(
        perp_exchange,
        symbol,
        int(start.value // 10**6),
        int(end.value // 10**6),
        timeframe="1m",
        tick_type="trade",
    )
    if frame is None or frame.empty:
        return pd.DataFrame()
    frame = frame.copy()
    frame.index = _normalise_datetime_index(frame.index, floor="min")
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    return frame


def compute_spread_relevant_candle_features(candles: pd.DataFrame) -> pd.DataFrame:
    if candles is None or candles.empty:
        return pd.DataFrame(columns=SPREAD_CANDLE_FEATURES)
    out = candles.copy()
    out.index = _normalise_datetime_index(out.index, floor="min")
    for col in ("open", "high", "low", "close", "volume"):
        out[col] = pd.to_numeric(out.get(col), errors="coerce")
    open_ = out["open"]
    high = out["high"]
    low = out["low"]
    close = out["close"]
    volume = out["volume"].clip(lower=0.0)
    quote_volume = (volume * close.abs()).clip(lower=0.0)
    denom_close = close.abs() + EPS
    range_abs = (high - low).clip(lower=0.0)
    body_high = pd.concat([open_, close], axis=1).max(axis=1)
    body_low = pd.concat([open_, close], axis=1).min(axis=1)
    upper = (high - body_high).clip(lower=0.0)
    lower = (body_low - low).clip(lower=0.0)
    prev_close = close.shift(1)
    features = pd.DataFrame(index=out.index)
    features["hl_range_bps"] = 10000.0 * range_abs / denom_close
    features["abs_return_bps"] = 10000.0 * (close / (open_.abs() + EPS) - 1.0).abs()
    features["body_bps"] = 10000.0 * (close - open_).abs() / denom_close
    features["upper_wick_bps"] = 10000.0 * upper / denom_close
    features["lower_wick_bps"] = 10000.0 * lower / denom_close
    features["wick_to_range"] = ((upper + lower) / (range_abs + EPS)).clip(0.0, 5.0)
    features["close_location"] = ((close - low) / (range_abs + EPS)).clip(0.0, 1.0)
    features["gap_bps"] = 10000.0 * (open_ - prev_close).abs() / (prev_close.abs() + EPS)
    features["log_candle_volume"] = np.log1p(volume)
    features["log_candle_quote_volume"] = np.log1p(quote_volume)
    return features.replace([np.inf, -np.inf], np.nan).astype(np.float32)


def add_spread_model_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    out.index = _normalise_datetime_index(out.index, floor="min")
    idx = pd.DatetimeIndex(out.index)

    def numeric(col: str, default: float = np.nan) -> pd.Series:
        if col in out.columns:
            return pd.to_numeric(out[col], errors="coerce")
        return pd.Series(default, index=out.index, dtype=np.float64)

    minute_of_day = idx.hour * 60 + idx.minute
    out["minute_of_day_sin"] = np.sin(2.0 * np.pi * minute_of_day / 1440.0)
    out["minute_of_day_cos"] = np.cos(2.0 * np.pi * minute_of_day / 1440.0)
    day_of_week = idx.dayofweek
    out["day_of_week_sin"] = np.sin(2.0 * np.pi * day_of_week / 7.0)
    out["day_of_week_cos"] = np.cos(2.0 * np.pi * day_of_week / 7.0)
    out["is_weekend"] = (day_of_week >= 5).astype(np.float64)

    rank_sources = {
        "candle_volume_rank": "log_candle_volume",
        "candle_quote_volume_rank": "log_candle_quote_volume",
        "hl_range_bps_rank": "hl_range_bps",
        "abs_return_bps_rank": "abs_return_bps",
        "body_bps_rank": "body_bps",
        "wick_to_range_rank": "wick_to_range",
        "gap_bps_rank": "gap_bps",
    }
    for feature, source in rank_sources.items():
        values = numeric(source)
        if values.notna().any():
            out[feature] = values.groupby(out.index).rank(pct=True, method="average")
        else:
            out[feature] = np.nan

    if "symbol" in out.columns:
        ordered = out.reset_index(names="_timestamp").reset_index(names="_row_order")
        ordered["_timestamp"] = pd.to_datetime(ordered["_timestamp"], utc=True, errors="coerce")
        ordered = ordered.sort_values(["symbol", "_timestamp", "_row_order"])
        grouped = ordered.groupby("symbol", sort=False, observed=False)
        rolling_sources = {
            "asset_hl_range_bps_roll_mean_3": "hl_range_bps",
            "asset_abs_return_bps_roll_mean_3": "abs_return_bps",
            "asset_log_candle_quote_volume_roll_mean_3": "log_candle_quote_volume",
        }
        for feature, source in rolling_sources.items():
            if source in ordered.columns:
                ordered[feature] = grouped[source].transform(
                    lambda x: pd.to_numeric(x, errors="coerce").rolling(3, min_periods=2).mean()
                )
            else:
                ordered[feature] = np.nan
        if "log_candle_quote_volume" in ordered.columns:
            ordered["asset_log_candle_quote_volume_lag1"] = grouped[
                "log_candle_quote_volume"
            ].shift(1)
        else:
            ordered["asset_log_candle_quote_volume_lag1"] = np.nan
        ordered = ordered.sort_values("_row_order").set_index("_timestamp")
        for feature in SPREAD_OHLCV_ROLLING_FEATURES:
            out[feature] = ordered[feature].to_numpy()
    else:
        for feature in SPREAD_OHLCV_ROLLING_FEATURES:
            out[feature] = np.nan

    for feature in SPREAD_MODEL_FEATURES:
        if feature not in out.columns:
            out[feature] = np.nan
    return out.replace([np.inf, -np.inf], np.nan)


def build_spread_training_frame(
    spreads: pd.DataFrame,
    candles: pd.DataFrame,
    *,
    symbol: Optional[str] = None,
) -> pd.DataFrame:
    if spreads is None or spreads.empty or candles is None or candles.empty:
        return pd.DataFrame()
    spread = spreads.copy()
    spread.index = _normalise_datetime_index(spread.index, floor="min")
    feats = compute_spread_relevant_candle_features(candles)
    joined = spread.join(feats, how="inner")
    if symbol is not None:
        joined["symbol"] = str(symbol)
    required = ["spread_bps", "spread_ticks", "min_tick_spread_bps", *SPREAD_CANDLE_FEATURES]
    for col in required:
        if col not in joined.columns:
            joined[col] = np.nan
    joined = joined.replace([np.inf, -np.inf], np.nan)
    joined = joined.dropna(subset=["spread_bps", "spread_ticks", *SPREAD_CANDLE_FEATURES])
    joined = joined[joined["spread_bps"] >= 0.0]
    return joined.sort_index()


def load_spread_snapshot_frame(path: Path | str) -> pd.DataFrame:
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(src)
    if src.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(src)
    elif src.suffix.lower() == ".json":
        frame = pd.read_json(src)
    else:
        frame = pd.read_csv(src)
    if "timestamp" in frame.columns:
        frame.index = pd.to_datetime(frame.pop("timestamp"), utc=True, errors="coerce")
    else:
        frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce")
    frame.index = pd.DatetimeIndex(frame.index).floor("min")
    return frame[~frame.index.isna()].sort_index()


def _safe_spearman(x: Sequence[float], y: Sequence[float]) -> float:
    xx = pd.Series(pd.to_numeric(pd.Series(x), errors="coerce"))
    yy = pd.Series(pd.to_numeric(pd.Series(y), errors="coerce"))
    mask = xx.notna() & yy.notna()
    if int(mask.sum()) < 5:
        return float("nan")
    if float(xx[mask].std()) <= EPS or float(yy[mask].std()) <= EPS:
        return float("nan")
    return float(xx[mask].rank().corr(yy[mask].rank()))


def compute_feature_ic_table(
    frame: pd.DataFrame,
    *,
    feature_names: Sequence[str] = SPREAD_MODEL_FEATURES,
    target_col: str = "spread_ticks",
) -> pd.DataFrame:
    rows = []
    for feat in feature_names:
        if feat not in frame.columns:
            continue
        ic = _safe_spearman(frame[feat], frame[target_col])
        rows.append(
            {
                "feature": str(feat),
                "ic": ic,
                "abs_ic": abs(ic) if np.isfinite(ic) else float("nan"),
                "n": int(pd.to_numeric(frame[feat], errors="coerce").notna().sum()),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["abs_ic", "feature"], ascending=[False, True]).reset_index(drop=True)


def compute_asset_spread_baseline(frame: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    if frame is None or frame.empty or "spread_bps" not in frame.columns:
        return pd.DataFrame(), 0.0
    work = frame.copy()
    if "symbol" not in work.columns:
        work["symbol"] = "__global__"
    work["spread_bps"] = pd.to_numeric(work["spread_bps"], errors="coerce").clip(lower=0.0)
    if "spread_ticks" in work.columns:
        work["spread_ticks"] = pd.to_numeric(work["spread_ticks"], errors="coerce")
    else:
        work["spread_ticks"] = np.nan
    valid = work.dropna(subset=["spread_bps"]).copy()
    if valid.empty:
        return pd.DataFrame(), 0.0
    global_baseline = float(valid["spread_bps"].mean())
    table = (
        valid.groupby("symbol", sort=True, observed=False)
        .agg(
            rows=("spread_bps", "size"),
            average_spread_bps=("spread_bps", "mean"),
            median_spread_bps=("spread_bps", "median"),
            p75_spread_bps=("spread_bps", lambda x: float(np.nanpercentile(x, 75))),
            average_spread_ticks=("spread_ticks", "mean"),
        )
        .reset_index()
    )
    return table, global_baseline


def _asset_baseline_lookup(artifact: Dict[str, Any]) -> Dict[str, float]:
    rows = artifact.get("per_asset_spread_baseline") or artifact.get("per_asset_average_spread") or []
    lookup: Dict[str, float] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol") or "").strip()
        if not symbol:
            continue
        value = row.get("average_spread_bps", row.get("baseline_spread_bps"))
        try:
            baseline = float(value)
        except Exception:
            continue
        if np.isfinite(baseline) and baseline >= 0.0:
            lookup[symbol] = baseline
    return lookup


def asset_baseline_spread_bps(
    frame: pd.DataFrame,
    artifact: Dict[str, Any],
) -> np.ndarray:
    default = artifact.get("global_average_spread_bps", artifact.get("baseline_spread_bps", 0.0))
    try:
        global_baseline = float(default)
    except Exception:
        global_baseline = 0.0
    if not np.isfinite(global_baseline) or global_baseline < 0.0:
        global_baseline = 0.0
    if frame is None or frame.empty:
        return np.asarray([], dtype=np.float64)
    if "symbol" not in frame.columns:
        return np.full(len(frame), global_baseline, dtype=np.float64)
    lookup = _asset_baseline_lookup(artifact)
    symbols = frame["symbol"].astype(str)
    baseline = symbols.map(lookup).fillna(global_baseline)
    return pd.to_numeric(baseline, errors="coerce").fillna(global_baseline).clip(lower=0.0).to_numpy(dtype=np.float64)


def _wide_classification_metrics(actual: np.ndarray, pred: np.ndarray, threshold: float) -> Dict[str, float]:
    y = actual >= float(threshold)
    p = pred >= float(threshold)
    tp = int(np.sum(y & p))
    fp = int(np.sum(~y & p))
    fn = int(np.sum(y & ~p))
    tn = int(np.sum(~y & ~p))
    out = {
        "wide_threshold_bps": float(threshold),
        "accuracy": float((tp + tn) / max(len(y), 1)),
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
        "f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
        "wide_actual_rate": float(np.mean(y)) if len(y) else 0.0,
        "wide_predicted_rate": float(np.mean(p)) if len(p) else 0.0,
    }
    if len(np.unique(y.astype(int))) == 2:
        try:
            out["auc"] = float(roc_auc_score(y.astype(int), pred))
        except Exception:
            out["auc"] = float("nan")
    else:
        out["auc"] = float("nan")
    return out


def _group_error(frame: pd.DataFrame, group_col: str) -> List[Dict[str, Any]]:
    if group_col not in frame.columns:
        return []
    rows: List[Dict[str, Any]] = []
    for key, grp in frame.groupby(group_col, sort=True, observed=False):
        err = pd.to_numeric(grp["prediction_error_bps"], errors="coerce")
        rows.append(
            {
                str(group_col): str(key),
                "rows": int(len(grp)),
                "mae_spread_bps": float(err.abs().mean()),
                "bias_pred_minus_actual_bps": float(err.mean()),
                "actual_spread_bps_mean": float(pd.to_numeric(grp["spread_bps"], errors="coerce").mean()),
                "predicted_spread_bps_mean": float(pd.to_numeric(grp["predicted_spread_bps"], errors="coerce").mean()),
            }
        )
    return rows


def _qcut_codes(values: Sequence[float], q: int) -> pd.Series:
    series = pd.Series(pd.to_numeric(pd.Series(values), errors="coerce"))
    if int(series.notna().sum()) < 2:
        return pd.Series(np.zeros(len(series), dtype=np.int16), index=series.index)
    try:
        return pd.qcut(
            series.rank(method="first"),
            min(int(q), int(series.notna().sum())),
            labels=False,
            duplicates="drop",
        )
    except Exception:
        return pd.Series(np.zeros(len(series), dtype=np.int16), index=series.index)


def predict_spread_bps(frame: pd.DataFrame, artifact: Dict[str, Any]) -> np.ndarray:
    features = list(artifact.get("selected_features", []))
    if not features:
        return np.full(len(frame), float("nan"), dtype=np.float64)
    prepared = add_spread_model_derived_features(frame)
    x = prepared.reindex(columns=features).apply(pd.to_numeric, errors="coerce")
    fill = artifact.get("feature_fill_values", {})
    for col in features:
        x[col] = x[col].fillna(float(fill.get(col, 0.0)))
    mean = np.asarray(artifact.get("scaler_mean", [0.0] * len(features)), dtype=np.float64)
    scale = np.asarray(artifact.get("scaler_scale", [1.0] * len(features)), dtype=np.float64)
    coef = np.asarray(artifact.get("ridge_coef", [0.0] * len(features)), dtype=np.float64)
    intercept = float(artifact.get("ridge_intercept", 0.0))
    scale = np.where(np.abs(scale) > EPS, scale, 1.0)
    values = x.to_numpy(dtype=np.float64)
    raw_pred = ((values - mean) / scale) @ coef + intercept
    target = str(artifact.get("target") or "")
    if "asset_average_spread_bps_baseline" in target or str(artifact.get("baseline_type") or "") == "per_asset_average_spread_bps":
        baseline = asset_baseline_spread_bps(prepared, artifact)
        pred = np.expm1(np.log1p(np.clip(baseline, 0.0, None)) + raw_pred)
    else:
        pred = np.expm1(raw_pred)
    min_tick = pd.to_numeric(
        prepared.get("min_tick_spread_bps", pd.Series(0.0, index=prepared.index)),
        errors="coerce",
    ).fillna(0.0).to_numpy(dtype=np.float64)
    return np.maximum(np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0), min_tick)


def evaluate_spread_predictions(frame: pd.DataFrame) -> Dict[str, Any]:
    if frame.empty:
        return {"rows": 0}
    actual = pd.to_numeric(frame["spread_bps"], errors="coerce").to_numpy(dtype=np.float64)
    pred = pd.to_numeric(frame["predicted_spread_bps"], errors="coerce").to_numpy(dtype=np.float64)
    mask = np.isfinite(actual) & np.isfinite(pred)
    if not mask.any():
        return {"rows": 0}
    actual = actual[mask]
    pred = pred[mask]
    eval_frame = frame.loc[mask].copy()
    eval_frame["prediction_error_bps"] = pred - actual
    eval_frame["actual_spread_decile"] = _qcut_codes(actual, 10).to_numpy()
    eval_frame["hour_utc"] = _normalise_datetime_index(eval_frame.index, floor="min").hour
    vol_source = pd.to_numeric(eval_frame.get("hl_range_bps"), errors="coerce")
    vol_codes = _qcut_codes(vol_source, 3).to_numpy()
    regime_names = np.asarray(["low", "medium", "high"], dtype=object)
    eval_frame["volatility_regime"] = regime_names[np.clip(vol_codes, 0, len(regime_names) - 1)]
    wide_threshold = float(np.nanpercentile(actual, 75))
    metrics = {
        "rows": int(len(actual)),
        "mae_spread_bps": float(np.mean(np.abs(pred - actual))),
        "mae_log1p_spread_bps": float(np.mean(np.abs(np.log1p(pred) - np.log1p(actual)))),
        "bias_pred_minus_actual_bps": float(np.mean(pred - actual)),
        "actual_spread_bps_mean": float(np.mean(actual)),
        "predicted_spread_bps_mean": float(np.mean(pred)),
        "predicted_spread_75th_percentile": float(np.nanpercentile(pred, 75)),
        "actual_spread_75th_percentile": wide_threshold,
        "error_by_spread_decile": _group_error(eval_frame, "actual_spread_decile"),
        "error_by_pair": _group_error(eval_frame, "symbol"),
        "error_by_time_of_day": _group_error(eval_frame, "hour_utc"),
        "error_by_volatility_regime": _group_error(eval_frame, "volatility_regime"),
        "wide_spread_classification": _wide_classification_metrics(actual, pred, wide_threshold),
    }
    if "asset_spread_baseline_bps" in eval_frame.columns:
        baseline = pd.to_numeric(eval_frame["asset_spread_baseline_bps"], errors="coerce").to_numpy(dtype=np.float64)
        baseline = np.nan_to_num(baseline, nan=float(np.nanmean(actual)), posinf=float(np.nanmean(actual)), neginf=0.0)
        baseline = np.clip(baseline, 0.0, None)
        baseline_mae = float(np.mean(np.abs(baseline - actual)))
        metrics.update(
            {
                "baseline_mae_spread_bps": baseline_mae,
                "baseline_mae_log1p_spread_bps": float(
                    np.mean(np.abs(np.log1p(baseline) - np.log1p(actual)))
                ),
                "baseline_bias_pred_minus_actual_bps": float(np.mean(baseline - actual)),
                "mae_improvement_vs_baseline_bps": float(baseline_mae - metrics["mae_spread_bps"]),
            }
        )
    return metrics


def fit_ridge_spread_model(
    frame: pd.DataFrame,
    *,
    top_k: int = 5,
    alpha: float = 1.0,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    if frame is None or frame.empty:
        raise ValueError("empty spread training frame")
    frame = add_spread_model_derived_features(frame)
    baseline_table, global_baseline = compute_asset_spread_baseline(frame)
    baseline_artifact = {
        "per_asset_spread_baseline": baseline_table.to_dict(orient="records"),
        "global_average_spread_bps": float(global_baseline),
    }
    actual_spread = pd.to_numeric(frame["spread_bps"], errors="coerce").clip(lower=0.0)
    baseline = asset_baseline_spread_bps(frame, baseline_artifact)
    frame["asset_spread_baseline_bps"] = baseline
    frame["spread_deviation_from_baseline_bps"] = actual_spread.to_numpy(dtype=np.float64) - baseline
    frame["spread_log1p_deviation_from_asset_average_baseline"] = (
        np.log1p(actual_spread.to_numpy(dtype=np.float64))
        - np.log1p(np.clip(baseline, 0.0, None))
    )
    ic_target = "spread_log1p_deviation_from_asset_average_baseline"
    ic_table = compute_feature_ic_table(frame, target_col=ic_target)
    selected = [
        str(row["feature"])
        for _, row in ic_table.head(int(top_k)).iterrows()
        if np.isfinite(float(row.get("abs_ic", np.nan)))
    ]
    if not selected:
        selected = list(SPREAD_MODEL_FEATURES[: int(top_k)])
    x = frame[selected].apply(pd.to_numeric, errors="coerce")
    fills = {col: float(x[col].median()) if x[col].notna().any() else 0.0 for col in selected}
    x = x.fillna(fills)
    y = pd.to_numeric(frame[ic_target], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x.to_numpy(dtype=np.float64))
    model = Ridge(alpha=float(alpha))
    model.fit(x_scaled, y)
    artifact = {
        "model_type": "kraken_spread_deviation_ridge_v3",
        "target": "log1p(spread_bps)-log1p(asset_average_spread_bps_baseline)",
        "baseline_type": "per_asset_average_spread_bps",
        "global_average_spread_bps": float(global_baseline),
        "per_asset_spread_baseline": baseline_table.to_dict(orient="records"),
        "prediction_transform": (
            "predicted_spread_bps=max("
            "expm1(log1p(asset_average_spread_bps_baseline)+y_pred), "
            "min_tick_spread_bps)"
        ),
        "feature_source": (
            "same_bar_1m_ohlcv_time_features_cross_sectional_ohlcv_ranks_"
            "and_rolling_ohlcv_summaries"
        ),
        "ic_target": ic_target,
        "candidate_features": list(SPREAD_MODEL_FEATURES),
        "selected_features": selected,
        "feature_ic_table": ic_table.to_dict(orient="records"),
        "ridge_alpha": float(alpha),
        "ridge_coef": [float(v) for v in model.coef_],
        "ridge_intercept": float(model.intercept_),
        "scaler_mean": [float(v) for v in scaler.mean_],
        "scaler_scale": [float(v if abs(v) > EPS else 1.0) for v in scaler.scale_],
        "feature_fill_values": fills,
    }
    scored = frame.copy()
    scored["predicted_spread_bps"] = predict_spread_bps(scored, artifact)
    scored["predicted_spread_deviation_from_baseline_bps"] = (
        scored["predicted_spread_bps"] - scored["asset_spread_baseline_bps"]
    )
    artifact["metrics"] = evaluate_spread_predictions(scored)
    asset_stats = (
        scored.groupby("symbol", sort=True)
        .agg(
            rows=("spread_bps", "size"),
            average_spread_bps=("spread_bps", "mean"),
            median_spread_bps=("spread_bps", "median"),
            p75_spread_bps=("spread_bps", lambda x: float(np.nanpercentile(x, 75))),
            predicted_p75_spread_bps=("predicted_spread_bps", lambda x: float(np.nanpercentile(x, 75))),
            average_spread_ticks=("spread_ticks", "mean"),
        )
        .reset_index()
    )
    artifact["per_asset_average_spread"] = asset_stats.to_dict(orient="records")
    artifact["predicted_spread_75th_percentile"] = float(
        artifact["metrics"].get("predicted_spread_75th_percentile", np.nan)
    )
    artifact["predicted_cost_bps"] = artifact["predicted_spread_75th_percentile"]
    return artifact, scored


def fetch_symbol_training_frame(
    item: SpreadUniverseItem,
    *,
    perp_exchange: Any,
    spreads: pd.DataFrame,
) -> pd.DataFrame:
    if spreads is None or spreads.empty or "symbol" not in spreads.columns:
        return pd.DataFrame()
    symbol_spreads = spreads[spreads["symbol"].astype(str) == item.perp_symbol].copy()
    if symbol_spreads.empty:
        return pd.DataFrame()
    candles = fetch_kraken_futures_1m_candles_for_spread_minutes(
        perp_exchange,
        item.perp_symbol,
        symbol_spreads.index,
    )
    if candles.empty:
        return pd.DataFrame()
    frame = build_spread_training_frame(symbol_spreads, candles, symbol=item.perp_symbol)
    frame["base"] = item.base
    frame["perp_market_id"] = item.perp_market_id
    frame["tick_size"] = float(item.tick_size)
    return frame


def collect_associated_candles_and_training(
    *,
    perp_exchange: Any,
    universe: Sequence[SpreadUniverseItem],
    spreads: pd.DataFrame,
    sleep_seconds: float = 0.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
    candle_frames: List[pd.DataFrame] = []
    training_frames: List[pd.DataFrame] = []
    audit: List[Dict[str, Any]] = []
    if spreads is None or spreads.empty or "symbol" not in spreads.columns:
        return pd.DataFrame(), pd.DataFrame(), audit
    for i, item in enumerate(universe, start=1):
        symbol_spreads = spreads[spreads["symbol"].astype(str) == item.perp_symbol].copy()
        if symbol_spreads.empty:
            audit.append({**asdict(item), "status": "no_spreads", "spread_rows": 0, "candle_rows": 0, "training_rows": 0})
            continue
        try:
            candles = fetch_kraken_futures_1m_candles_for_spread_minutes(
                perp_exchange,
                item.perp_symbol,
                symbol_spreads.index,
            )
            candle_rows = int(len(candles))
            if not candles.empty:
                candles_out = candles.copy()
                candles_out["symbol"] = item.perp_symbol
                candles_out["base"] = item.base
                candles_out["perp_market_id"] = item.perp_market_id
                candle_frames.append(candles_out)
                training = build_spread_training_frame(
                    symbol_spreads,
                    candles,
                    symbol=item.perp_symbol,
                )
                training["base"] = item.base
                training["perp_market_id"] = item.perp_market_id
                training["tick_size"] = float(item.tick_size)
            else:
                training = pd.DataFrame()
            training_rows = int(len(training))
            if not training.empty:
                training_frames.append(training)
            status = "ok" if candle_rows > 0 else "empty_candles"
        except Exception as exc:
            candle_rows = 0
            training_rows = 0
            status = f"failed:{exc.__class__.__name__}:{exc}"
        audit.append(
            {
                **asdict(item),
                "status": status,
                "spread_rows": int(len(symbol_spreads)),
                "candle_rows": candle_rows,
                "training_rows": training_rows,
            }
        )
        tprint(
            f"[{i:04d}/{len(universe):04d}] {item.perp_symbol} "
            f"spread rows={len(symbol_spreads)} candle rows={candle_rows} "
            f"training rows={training_rows} status={status}"
        )
        if sleep_seconds > 0.0:
            time.sleep(float(sleep_seconds))
    candles_all = pd.concat(candle_frames, axis=0).sort_index() if candle_frames else pd.DataFrame()
    training_all = pd.concat(training_frames, axis=0).sort_index() if training_frames else pd.DataFrame()
    return candles_all, training_all, audit


def train_kraken_spread_model(
    *,
    lookback_hours: float = 24.0,
    symbols: Optional[Iterable[str]] = None,
    max_symbols: int = 0,
    sleep_seconds: float = 0.25,
    spread_snapshot_path: Optional[Path | str] = None,
    training_frame_path: Optional[Path | str] = None,
    snapshot_count: int = 1,
    snapshot_interval_seconds: float = 0.0,
    top_k: int = 5,
    alpha: float = 1.0,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    since_ts = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=float(lookback_hours))
    if training_frame_path:
        training = load_spread_snapshot_frame(training_frame_path)
        if lookback_hours and float(lookback_hours) > 0.0:
            training = training[training.index >= since_ts.floor("min")]
        if symbols:
            allowed = {str(s).upper().strip() for s in symbols if str(s).strip()}
            if allowed and "symbol" in training.columns:
                training = training[training["symbol"].astype(str).str.upper().isin(allowed)]
        if max_symbols and int(max_symbols) > 0 and "symbol" in training.columns:
            keep_symbols = sorted(training["symbol"].astype(str).unique())[: int(max_symbols)]
            training = training[training["symbol"].astype(str).isin(keep_symbols)]
        if training.empty:
            raise RuntimeError("No spread/candle training rows were loaded")
        artifact, scored = fit_ridge_spread_model(training, top_k=top_k, alpha=alpha)
        if "symbol" in training.columns:
            fetch_audit = (
                training.groupby("symbol", sort=True)
                .size()
                .reset_index(name="rows")
                .assign(status="loaded_joined_training_frame")
            )
        else:
            fetch_audit = pd.DataFrame(
                [{"symbol": "__all__", "rows": int(len(training)), "status": "loaded_joined_training_frame"}]
            )
        artifact.update(
            {
                "lookback_hours": float(lookback_hours),
                "since_ts": since_ts.isoformat(),
                "created_ts": pd.Timestamp.now(tz="UTC").isoformat(),
                "universe_count": int(fetch_audit["symbol"].nunique()) if "symbol" in fetch_audit.columns else 0,
                "fetched_symbol_count": int(fetch_audit["rows"].gt(0).sum()) if "rows" in fetch_audit.columns else 0,
                "fetch_audit": fetch_audit.to_dict(orient="records"),
                "universe_audit": [],
                "spread_data_source": "loaded_joined_training_frame",
                "source_endpoint": KRAKEN_FUTURES_TICKERS_URL,
                "spread_snapshot_path": str(spread_snapshot_path) if spread_snapshot_path else None,
                "training_frame_path": str(training_frame_path),
                "snapshot_count": int(snapshot_count),
                "snapshot_interval_seconds": float(snapshot_interval_seconds),
            }
        )
        return artifact, scored, fetch_audit

    perp_exchange = make_perp_exchange()
    universe, audit = resolve_spread_universe(
        perp_exchange,
        symbols=symbols,
    )
    if max_symbols and int(max_symbols) > 0:
        universe = universe[: int(max_symbols)]
    if spread_snapshot_path:
        spread_rows = load_spread_snapshot_frame(spread_snapshot_path)
        if lookback_hours and float(lookback_hours) > 0.0:
            spread_rows = spread_rows[spread_rows.index >= since_ts.floor("min")]
        source_mode = "loaded_futures_ticker_snapshots"
    else:
        spread_rows = collect_kraken_futures_ticker_spreads(
            universe=universe,
            snapshot_count=int(snapshot_count),
            snapshot_interval_seconds=float(snapshot_interval_seconds),
        )
        source_mode = "live_futures_ticker_snapshots"
    frames: List[pd.DataFrame] = []
    fetch_rows: List[Dict[str, Any]] = []
    for i, item in enumerate(universe, start=1):
        try:
            frame = fetch_symbol_training_frame(
                item,
                perp_exchange=perp_exchange,
                spreads=spread_rows,
            )
            status = "ok" if not frame.empty else "empty"
            rows = int(len(frame))
            if not frame.empty:
                frames.append(frame)
        except Exception as exc:
            frame = pd.DataFrame()
            status = f"failed:{exc.__class__.__name__}:{exc}"
            rows = 0
        fetch_rows.append({**asdict(item), "status": status, "rows": rows})
        tprint(f"[{i:04d}/{len(universe):04d}] {item.perp_symbol} spread rows={rows} status={status}")
        if sleep_seconds > 0.0:
            time.sleep(float(sleep_seconds))
    if not frames:
        raise RuntimeError("No spread/candle training rows were fetched")
    training = pd.concat(frames, axis=0).sort_index()
    artifact, scored = fit_ridge_spread_model(training, top_k=top_k, alpha=alpha)
    artifact.update(
        {
            "lookback_hours": float(lookback_hours),
            "since_ts": since_ts.isoformat(),
            "created_ts": pd.Timestamp.now(tz="UTC").isoformat(),
            "universe_count": int(len(universe)),
            "fetched_symbol_count": int(sum(1 for row in fetch_rows if row["rows"] > 0)),
            "fetch_audit": fetch_rows,
            "universe_audit": audit,
            "spread_data_source": source_mode,
            "source_endpoint": KRAKEN_FUTURES_TICKERS_URL,
            "spread_snapshot_path": str(spread_snapshot_path) if spread_snapshot_path else None,
            "training_frame_path": None,
            "snapshot_count": int(snapshot_count),
            "snapshot_interval_seconds": float(snapshot_interval_seconds),
        }
    )
    return artifact, scored, pd.DataFrame(fetch_rows)


def save_spread_model_outputs(
    artifact: Dict[str, Any],
    scored: pd.DataFrame,
    fetch_audit: pd.DataFrame,
    *,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scored_path = out_dir / "spread_training_scored_latest.parquet"
    audit_path = out_dir / "spread_fetch_audit_latest.csv"
    baseline_path = out_dir / "per_asset_spread_baseline_latest.csv"
    artifact_path = out_dir / "spread_model_latest.json"
    scored.to_parquet(scored_path, compression="zstd")
    fetch_audit.to_csv(audit_path, index=False)
    baseline_rows = artifact.get("per_asset_spread_baseline") or artifact.get("per_asset_average_spread") or []
    pd.DataFrame(baseline_rows).to_csv(baseline_path, index=False)
    serializable = _jsonify(artifact)
    serializable["scored_rows_path"] = str(scored_path.resolve())
    serializable["fetch_audit_path"] = str(audit_path.resolve())
    serializable["per_asset_spread_baseline_path"] = str(baseline_path.resolve())
    tmp = artifact_path.with_suffix(".tmp.json")
    tmp.write_text(
        json.dumps(serializable, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(artifact_path)
    return artifact_path


def save_spread_baseline_outputs(
    frame: pd.DataFrame,
    *,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> Tuple[Path, Path, Dict[str, Any]]:
    if frame is None or frame.empty:
        raise ValueError("empty spread baseline frame")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_table, global_baseline = compute_asset_spread_baseline(frame)
    if baseline_table.empty:
        raise ValueError("no valid spread rows for baseline")
    baseline_path = out_dir / "per_asset_spread_baseline_latest.csv"
    summary_path = out_dir / "per_asset_spread_baseline_latest.json"
    baseline_table.to_csv(baseline_path, index=False)
    idx = _normalise_datetime_index(frame.index, floor="min")
    payload = {
        "generated_by": "kraken_spread_model",
        "schema": "per_asset_spread_baseline_v1",
        "created_ts": pd.Timestamp.now(tz="UTC").isoformat(),
        "rows": int(len(frame)),
        "symbols": int(baseline_table["symbol"].nunique()),
        "timestamp_min": idx.min().isoformat() if len(idx) else None,
        "timestamp_max": idx.max().isoformat() if len(idx) else None,
        "global_average_spread_bps": float(global_baseline),
        "baseline_path": str(baseline_path.resolve()),
        "per_asset_spread_baseline": baseline_table.to_dict(orient="records"),
    }
    summary_path.write_text(
        json.dumps(_jsonify(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return baseline_path, summary_path, payload


def _jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonify(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonify(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonify(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def load_spread_cost_bps(path: Path | str = DEFAULT_OUTPUT_DIR / "spread_model_latest.json") -> Optional[float]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        value = float(payload.get("predicted_cost_bps", payload.get("predicted_spread_75th_percentile")))
    except Exception:
        return None
    return value if np.isfinite(value) and value >= 0.0 else None


def collect_spread_snapshots_main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Collect Kraken Futures perp L1 bid/ask spread snapshots."
    )
    parser.add_argument("--symbols", default="", help="Comma-separated base or perp symbols.")
    parser.add_argument("--max-symbols", type=int, default=0)
    parser.add_argument("--snapshot-count", type=int, default=1)
    parser.add_argument("--snapshot-interval-seconds", type=float, default=0.0)
    parser.add_argument("--candle-sleep-seconds", type=float, default=0.25)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--cycle-sleep-seconds", type=float, default=0.0)
    parser.add_argument(
        "--hourly-top-of-hour",
        action="store_true",
        help="Align each collection cycle to the beginning of the UTC hour and collect one snapshot minute.",
    )
    parser.add_argument(
        "--top-of-hour-grace-seconds",
        type=float,
        default=5.0,
        help="Treat script starts within this many seconds after the hour as on-time.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_SNAPSHOT_OUTPUT_DIR))
    args = parser.parse_args(argv)

    symbols = [s.strip() for s in str(args.symbols).split(",") if s.strip()] or None
    perp_exchange = make_perp_exchange()
    universe, audit = resolve_spread_universe(perp_exchange, symbols=symbols)
    if args.max_symbols and int(args.max_symbols) > 0:
        universe = universe[: int(args.max_symbols)]
        allowed = {item.perp_symbol for item in universe}
        audit = [row for row in audit if str(row.get("perp_symbol")) in allowed]
    if not universe:
        raise RuntimeError("No eligible Kraken Futures perp markets found for spread collection")
    successes = 0
    requested_cycles = int(args.cycles)
    run_forever = requested_cycles <= 0
    cycles = requested_cycles if requested_cycles > 0 else 1
    cycle = 0
    last_hourly_attempt: Optional[pd.Timestamp] = None
    while run_forever or cycle < cycles:
        cycle += 1
        snapshot_count = 1 if bool(args.hourly_top_of_hour) else int(args.snapshot_count)
        snapshot_interval_seconds = (
            0.0
            if bool(args.hourly_top_of_hour)
            else float(args.snapshot_interval_seconds)
        )
        if bool(args.hourly_top_of_hour):
            while True:
                now_ts = pd.Timestamp.now(tz="UTC")
                hour_start = now_ts.floor("h")
                target_hour = hour_start
                wait_seconds = _seconds_until_next_hour(
                    now_ts,
                    grace_seconds=float(args.top_of_hour_grace_seconds),
                )
                if wait_seconds > 0.0:
                    target_hour = hour_start + pd.Timedelta(hours=1)
                if last_hourly_attempt is not None and hour_start <= last_hourly_attempt:
                    target_hour = last_hourly_attempt + pd.Timedelta(hours=1)
                    wait_seconds = max(float((target_hour - now_ts).total_seconds()), 0.0)
                if wait_seconds <= 0.0:
                    break
                print(
                    json.dumps(
                        {
                            "cycle": int(cycle),
                            "cycles": None if run_forever else int(cycles),
                            "run_forever": bool(run_forever),
                            "status": "waiting_for_top_of_hour",
                            "sleep_seconds": float(wait_seconds),
                            "target_hour_utc": target_hour.isoformat(),
                            "hourly_top_of_hour": True,
                        },
                        indent=2,
                        sort_keys=True,
                    ),
                    flush=True,
                )
                time.sleep(float(wait_seconds))
            last_hourly_attempt = pd.Timestamp.now(tz="UTC").floor("h")
        try:
            frame = collect_kraken_futures_ticker_spreads(
                universe=universe,
                snapshot_count=int(snapshot_count),
                snapshot_interval_seconds=float(snapshot_interval_seconds),
            )
            if frame.empty:
                raise RuntimeError("No Kraken Futures ticker spread snapshots were collected")
            candles, training, candle_audit = collect_associated_candles_and_training(
                perp_exchange=perp_exchange,
                universe=universe,
                spreads=frame,
                sleep_seconds=float(args.candle_sleep_seconds),
            )
            if candles.empty:
                raise RuntimeError("No associated Kraken Futures 1m candles were collected")
            parquet_path, summary_path = save_spread_snapshot_collection(
                frame,
                universe_audit=audit,
                candles=candles,
                training=training,
                candle_audit=candle_audit,
                output_dir=args.output_dir,
            )
            summary = _spread_snapshot_summary(frame, universe_count=len(audit))
            successes += 1
            print(
                json.dumps(
                    {
                        "cycle": int(cycle),
                        "cycles": None if run_forever else int(cycles),
                        "run_forever": bool(run_forever),
                        "hourly_top_of_hour": bool(args.hourly_top_of_hour),
                        "snapshot_count": int(snapshot_count),
                        "snapshot_interval_seconds": float(snapshot_interval_seconds),
                        "snapshot_path": str(parquet_path),
                        "summary_path": str(summary_path),
                        "latest_snapshot_path": str(Path(args.output_dir) / "latest.parquet"),
                        "latest_candles_path": str(Path(args.output_dir) / "latest_candles.parquet"),
                        "latest_training_path": str(Path(args.output_dir) / "latest_training.parquet"),
                        "history_snapshot_path": str(Path(args.output_dir) / "history.parquet"),
                        "history_candles_path": str(Path(args.output_dir) / "history_candles.parquet"),
                        "history_training_path": str(Path(args.output_dir) / "history_training.parquet"),
                        "rows": summary.get("rows"),
                        "symbols": summary.get("symbols"),
                        "candle_rows": int(len(candles)),
                        "training_rows": int(len(training)),
                        "timestamp_min": summary.get("timestamp_min"),
                        "timestamp_max": summary.get("timestamp_max"),
                        "spread_bps_p75": summary.get("spread_bps_p75"),
                    },
                    indent=2,
                    sort_keys=True,
                ),
                flush=True,
            )
        except Exception as exc:
            payload = {
                "cycle": int(cycle),
                "cycles": None if run_forever else int(cycles),
                "run_forever": bool(run_forever),
                "hourly_top_of_hour": bool(args.hourly_top_of_hour),
                "status": "failed",
                "error": f"{exc.__class__.__name__}: {exc}",
            }
            print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
            if not run_forever and cycles == 1:
                raise
        if bool(args.hourly_top_of_hour):
            continue
        if cycle < cycles and float(args.cycle_sleep_seconds) > 0.0:
            time.sleep(float(args.cycle_sleep_seconds))
    if successes <= 0:
        raise RuntimeError("No spread snapshot collection cycles completed successfully")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Kraken perps spread prediction model from Kraken Futures "
            "L1 bid/ask snapshots and matching 1m perp candles."
        )
    )
    parser.add_argument("--lookback-hours", type=float, default=24.0)
    parser.add_argument("--symbols", default="", help="Comma-separated base or perp symbols.")
    parser.add_argument("--max-symbols", type=int, default=0)
    parser.add_argument("--sleep-seconds", type=float, default=0.25)
    parser.add_argument(
        "--spread-snapshot-path",
        default="",
        help="CSV/Parquet/JSON file of collected Kraken Futures bid/ask snapshots.",
    )
    parser.add_argument(
        "--training-frame-path",
        default="",
        help="CSV/Parquet/JSON file of already joined spread/candle training rows.",
    )
    parser.add_argument("--snapshot-count", type=int, default=1)
    parser.add_argument("--snapshot-interval-seconds", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Only update per-asset average spread baseline artifacts; do not train Ridge.",
    )
    args = parser.parse_args(argv)

    symbols = [s.strip() for s in str(args.symbols).split(",") if s.strip()] or None
    if bool(args.baseline_only):
        training_path = str(args.training_frame_path).strip()
        if not training_path:
            raise ValueError("--baseline-only requires --training-frame-path")
        frame = load_spread_snapshot_frame(training_path)
        if args.lookback_hours and float(args.lookback_hours) > 0.0:
            since_ts = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=float(args.lookback_hours))
            frame = frame[frame.index >= since_ts.floor("min")]
        if symbols and "symbol" in frame.columns:
            allowed = {str(s).upper().strip() for s in symbols if str(s).strip()}
            frame = frame[frame["symbol"].astype(str).str.upper().isin(allowed)]
        if args.max_symbols and int(args.max_symbols) > 0 and "symbol" in frame.columns:
            keep_symbols = sorted(frame["symbol"].astype(str).unique())[: int(args.max_symbols)]
            frame = frame[frame["symbol"].astype(str).isin(keep_symbols)]
        baseline_path, summary_path, summary = save_spread_baseline_outputs(
            frame,
            output_dir=args.output_dir,
        )
        print(
            json.dumps(
                {
                    "baseline_path": str(baseline_path),
                    "summary_path": str(summary_path),
                    "rows": summary.get("rows"),
                    "symbols": summary.get("symbols"),
                    "timestamp_min": summary.get("timestamp_min"),
                    "timestamp_max": summary.get("timestamp_max"),
                    "global_average_spread_bps": summary.get("global_average_spread_bps"),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    artifact, scored, fetch_audit = train_kraken_spread_model(
        lookback_hours=float(args.lookback_hours),
        symbols=symbols,
        max_symbols=int(args.max_symbols),
        sleep_seconds=float(args.sleep_seconds),
        spread_snapshot_path=str(args.spread_snapshot_path).strip() or None,
        training_frame_path=str(args.training_frame_path).strip() or None,
        snapshot_count=int(args.snapshot_count),
        snapshot_interval_seconds=float(args.snapshot_interval_seconds),
        top_k=int(args.top_k),
        alpha=float(args.ridge_alpha),
    )
    artifact_path = save_spread_model_outputs(
        artifact,
        scored,
        fetch_audit,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "artifact_path": str(artifact_path),
                "rows": int(len(scored)),
                "symbols": int(fetch_audit["rows"].gt(0).sum()) if not fetch_audit.empty else 0,
                "selected_features": artifact.get("selected_features", []),
                "predicted_cost_bps": artifact.get("predicted_cost_bps"),
                "mae_spread_bps": (artifact.get("metrics") or {}).get("mae_spread_bps"),
                "baseline_mae_spread_bps": (artifact.get("metrics") or {}).get("baseline_mae_spread_bps"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
