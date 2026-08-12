#!/usr/bin/env python3
"""Materialise a declared 15-minute proxy for the TP6/SL4/H12 R3 contract.

The source panel supplies decision-time features and the causal ATR.  Future
path values come only from the local 15-minute OHLCV store.  This is a
development label source, not an exact-minute execution replay: every output
row carries ``label_resolution=proxy_15m`` and the manifest records the
entry/path convention and cost-once rule.

The implementation works part-by-part and symbol-by-symbol so a complete
current-year panel can be materialised without loading the full OHLCV universe
or a dense all-symbol path tensor into memory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SIDES = ("long", "short")
COST_BPS = 100.0
TP_ATR = 6.0
SL_ATR = 4.0
HORIZON_BARS = 48
ENTRY_DELAY_HOURS = 1
BAR_MINUTES = 15
HURDLE_BPS = 25.0
SOFT_TEMPERATURE_BPS = 50.0


def _symbol_file(symbol: str, source: Path) -> Path:
    return source / f"{str(symbol).lower().replace('/', '')}_15m.parquet"


def _load_ohlcv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.read_parquet(path, columns=["open", "high", "low", "close"])
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError(f"15m source index is not datetime: {path}")
    idx = pd.DatetimeIndex(frame.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    order = np.argsort(idx.asi8)
    ts = idx.asi8[order].astype(np.int64, copy=False)
    # Duplicate timestamps cannot be used for an unambiguous bar path.
    if len(ts) != len(np.unique(ts)):
        keep = ~pd.Index(ts).duplicated(keep="last")
        order = order[keep]
        ts = ts[keep]
    return (
        ts,
        frame.open.to_numpy(np.float64, copy=False)[order],
        frame.high.to_numpy(np.float64, copy=False)[order],
        frame.low.to_numpy(np.float64, copy=False)[order],
        frame.close.to_numpy(np.float64, copy=False)[order],
    )


def _path_labels(
    rows: pd.DataFrame,
    *,
    ts: np.ndarray,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
) -> pd.DataFrame:
    """Return proxy labels for one symbol in a vectorised chunk."""
    decision_ns = pd.to_datetime(rows["__ts__"], utc=True).astype("int64").to_numpy()
    entry_ns = decision_ns + ENTRY_DELAY_HOURS * 3_600_000_000_000
    pos = np.searchsorted(ts, entry_ns, side="left")
    valid_entry = (pos < len(ts)) & (ts[np.minimum(pos, max(len(ts) - 1, 0))] == entry_ns)
    # A complete, contiguous path is required.  This prevents a data gap from
    # becoming a fabricated timeout.
    valid_path = valid_entry.copy()
    if len(ts):
        safe_pos = np.minimum(pos, len(ts) - 1)
        for step in range(HORIZON_BARS):
            valid_path &= (safe_pos + step < len(ts))
            if np.any(valid_path):
                wanted = entry_ns + step * BAR_MINUTES * 60 * 1_000_000_000
                valid_path &= (safe_pos + step < len(ts)) & (ts[np.minimum(safe_pos + step, len(ts) - 1)] == wanted)

    n = len(rows)
    gross = np.full(n, np.nan, dtype=np.float64)
    net = np.full(n, np.nan, dtype=np.float64)
    pre_mfe = np.full(n, np.nan, dtype=np.float64)
    lower_minute = np.full(n, -1, dtype=np.int16)
    event = np.full(n, -1, dtype=np.int8)
    atr = pd.to_numeric(rows["atr_bps"], errors="coerce").to_numpy(np.float64)
    valid_path &= np.isfinite(atr) & (atr > 0.0)

    valid_idx = np.flatnonzero(valid_path)
    if len(valid_idx):
        p = pos[valid_idx]
        offsets = np.arange(HORIZON_BARS, dtype=np.int64)[None, :]
        path_pos = p[:, None] + offsets
        entry = opens[p]
        path_high = highs[path_pos]
        path_low = lows[path_pos]
        path_close = closes[path_pos]
        side = rows.side_name.iloc[valid_idx].astype(str).to_numpy()
        a = atr[valid_idx]
        tp_ret = TP_ATR * a / 10_000.0
        sl_ret = SL_ATR * a / 10_000.0
        long_mask = side == "long"
        upper_price = np.where(long_mask, entry * (1.0 + tp_ret), entry * (1.0 - tp_ret))
        lower_price = np.where(long_mask, entry * (1.0 - sl_ret), entry * (1.0 + sl_ret))
        upper_hit = np.where(long_mask[:, None], path_high >= upper_price[:, None], path_low <= upper_price[:, None])
        lower_hit = np.where(long_mask[:, None], path_low <= lower_price[:, None], path_high >= lower_price[:, None])
        has_upper = upper_hit.any(axis=1)
        has_lower = lower_hit.any(axis=1)
        upper_at = np.where(has_upper, upper_hit.argmax(axis=1), HORIZON_BARS)
        lower_at = np.where(has_lower, lower_hit.argmax(axis=1), HORIZON_BARS)
        # A missing touch uses the horizon sentinel for both indices.  Do
        # not classify that collision as adverse: it is a genuine timeout.
        # The explicit ``has_*`` guards also keep the first-touch semantics
        # unambiguous when only one barrier is reached.
        adverse_first = has_lower & (lower_at <= upper_at)
        upper_first = has_upper & (upper_at < lower_at)
        timeout = ~(adverse_first | upper_first)
        event_values = np.where(adverse_first, 0, np.where(timeout, 1, 2)).astype(np.int8)
        event[valid_idx] = event_values
        lower_minute[valid_idx] = np.where(has_lower, lower_at * BAR_MINUTES, -1).astype(np.int16)

        favorable = np.where(long_mask[:, None], (path_high / entry[:, None] - 1.0) * 10_000.0, (1.0 - path_low / entry[:, None]) * 10_000.0)
        # The adverse bar itself is excluded from pre-adverse MFE.  For a
        # timeout or upper-first path, all bars up to the selected horizon are
        # available to the opportunity target.
        cutoff = np.where(adverse_first, lower_at, HORIZON_BARS)
        before = np.arange(HORIZON_BARS)[None, :] < cutoff[:, None]
        # Use -inf instead of an all-NaN reduction.  A path that reaches the
        # adverse barrier in its first bar has no pre-adverse excursion and is
        # assigned zero below.
        pre_mfe_values = np.where(before, favorable, -np.inf)
        pre_mfe_values = np.max(pre_mfe_values, axis=1)
        pre_mfe[valid_idx] = np.where(np.isfinite(pre_mfe_values), pre_mfe_values, 0.0)

        terminal = (path_close[np.arange(len(valid_idx)), np.minimum(np.maximum(upper_at, lower_at), HORIZON_BARS - 1)] / entry - 1.0) * 10_000.0
        upper_gross = np.where(long_mask, tp_ret * 10_000.0, tp_ret * 10_000.0)
        lower_gross = -sl_ret * 10_000.0
        gross_values = np.where(upper_first, upper_gross, np.where(adverse_first, lower_gross, terminal))
        gross[valid_idx] = gross_values
        net[valid_idx] = gross_values - COST_BPS

    robust = np.where(valid_path, (pre_mfe - COST_BPS - HURDLE_BPS > 0.0).astype(np.int8), np.nan)
    # A valid adverse-first path is adverse even if it briefly had a positive
    # excursion; R3 is a competing-risk opportunity target.
    robust = np.where(valid_path & (event == 0), 0.0, robust)
    r3_class = np.where(valid_path, np.where(robust == 1.0, 2, np.where(event == 0, 0, 1)), np.nan)
    soft = np.where(valid_path, 1.0 / (1.0 + np.exp(-np.clip((pre_mfe - COST_BPS - HURDLE_BPS) / SOFT_TEMPERATURE_BPS, -50.0, 50.0))), np.nan)
    out = pd.DataFrame({
        "label_valid": valid_path,
        "proxy_entry_valid": valid_entry,
        "proxy_path_complete": valid_path,
        "gross_bps_proxy_15m": gross,
        "net_bps_proxy_15m": net,
        "pre_adverse_mfe_bps_proxy_15m": pre_mfe,
        "lower_touch_minute_proxy_15m": lower_minute,
        "t2_tp6_sl4_event_proxy_15m": event,
        "robust_clear_event_b25_proxy_15m": robust,
        "robust_clear_soft_b25_t50_proxy_15m": soft,
        "r3_class_proxy_15m": r3_class,
        "label_available_ts_proxy_15m": pd.to_datetime(rows["__ts__"], utc=True) + pd.Timedelta(hours=13),
        "label_resolution": "proxy_15m",
    }, index=rows.index)
    return out


def _iter_parts(root: Path, months: Iterable[str] | None) -> list[Path]:
    paths = sorted(root.glob("month=*/part-*.parquet"))
    if months is None:
        return paths
    allowed = set(months)
    return [p for p in paths if p.parent.name.split("=", 1)[-1] in allowed]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True, help="partitioned panel with __ts__, __symbol__, side_name and atr_bps")
    parser.add_argument("--ohlcv-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--months", nargs="*", default=None)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    rows_total = valid_total = 0
    source_symbols: set[str] = set()
    part_manifests: list[dict[str, object]] = []
    cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for source_part in _iter_parts(args.panel, args.months):
        month = source_part.parent.name.split("=", 1)[-1]
        source = pd.read_parquet(source_part)
        required = {"candidate_id", "__ts__", "__symbol__", "side_name", "atr_bps"}
        missing = sorted(required.difference(source.columns))
        if missing:
            raise ValueError(f"{source_part} missing {missing}")
        if source.candidate_id.duplicated().any():
            raise ValueError(f"duplicate candidate IDs in {source_part}")
        pieces: list[pd.DataFrame] = []
        for symbol, group in source.groupby("__symbol__", sort=False, observed=True):
            symbol = str(symbol)
            path = _symbol_file(symbol, args.ohlcv_root)
            if not path.exists():
                raise FileNotFoundError(f"15m source missing for {symbol}: {path}")
            if symbol not in cache:
                cache[symbol] = _load_ohlcv(path)
            ts, opens, highs, lows, closes = cache[symbol]
            pieces.append(_path_labels(group, ts=ts, opens=opens, highs=highs, lows=lows, closes=closes))
            source_symbols.add(symbol)
        labels = pd.concat(pieces).sort_index()
        output = source.reset_index(drop=True).copy()
        labels = labels.reindex(output.index)
        exact_net_source = pd.to_numeric(output.get("exact_net_bps"), errors="coerce") if "exact_net_bps" in output else pd.Series(np.nan, index=output.index)
        # Keep only the causal input contract plus identity and proxy labels;
        # exact outcome/path columns in the source panel are retained under
        # explicit ``exact_*`` names for overlap diagnostics, never as model
        # inputs.
        feature_columns = [c for c in output.columns if c not in {
            "label_available_ts", "label_valid", "exact_gross_bps", "exact_net_bps",
            "r3_class", "robust_clear_event_b0", "robust_clear_event_b25", "robust_clear_event_b50",
            "robust_clear_soft_b25_t50", "lower_touch_minute", "t2_tp6_sl4_event",
            "pre_adverse_mfe_bps", "pre_adverse_mfe_atr",
        }]
        output = output.loc[:, feature_columns].copy()
        output["decision_ts"] = pd.to_datetime(output["__ts__"], utc=True)
        for column in labels.columns:
            output[column] = labels[column].to_numpy()
        output["exact_overlap_valid"] = output["label_valid"].astype(bool) & exact_net_source.notna()
        output = output.drop(columns=["__ts__"], errors="ignore")
        target = args.out / "parts" / f"month={month}" / source_part.name
        target.parent.mkdir(parents=True, exist_ok=True)
        output.to_parquet(target, index=False, compression="zstd")
        rows_total += len(output)
        valid_total += int(output.label_valid.sum())
        part_manifests.append({"source": str(source_part), "output": str(target), "month": month, "rows": len(output), "valid_rows": int(output.label_valid.sum())})
        print(json.dumps(part_manifests[-1]), flush=True)
    coverage = pd.DataFrame(part_manifests)
    coverage.to_parquet(args.out / "coverage.parquet", index=False)
    manifest = {
        "schema": "tp6_sl4_h12_r3_proxy_15m_v1",
        "status": "complete",
        "rows": rows_total,
        "valid_rows": valid_total,
        "valid_rate": valid_total / rows_total if rows_total else 0.0,
        "source_symbols": len(source_symbols),
        "entry": "decision timestamp + 1h, first 15m bar at the declared timestamp",
        "path": "48 contiguous 15m bars (12h), TP +6 ATR / SL -4 ATR, adverse same-bar precedence",
        "atr": "causal decision-time atr_bps supplied by the feature panel",
        "cost_bps": COST_BPS,
        "r3": "pre-adverse MFE - 100 - 25 > 0; sigmoid temperature 50 bps",
        "invalid_rows": "targets null; retained for coverage only",
        "label_resolution": "proxy_15m",
        "parts": part_manifests,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("schema", "status", "rows", "valid_rows", "valid_rate", "source_symbols")}, indent=2))


if __name__ == "__main__":
    main()
