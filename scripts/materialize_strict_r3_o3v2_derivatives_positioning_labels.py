#!/usr/bin/env python3
"""Add resolved H12 aggregate OI/funding positioning labels to a timestamp sidecar.

The input is an existing timestamp-level market-label sidecar.  OI and funding
are read only from the historical Kraken Futures hourly source panels.  Funding
may be forward-filled *from an already observed earlier print*, for at most
12 hours; it is never backward-filled.  The resulting target table remains
labels-only and cannot be used in scoring/inference.

There is no historical forced-liquidation tape in this source contract.  The
third label is consequently named ``market_liquidation_imbalance_proxy_12h``:
it is an explicitly declared crowding-weighted OI-deleveraging proxy, not a
claim of observed exchange liquidation flow.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "strict_r3_o3v2_derivatives_positioning_timestamp_labels_v1"
HOURS = 12
MIN_OI_ASSETS = 50
MIN_FUNDING_ASSETS = 25
DEFAULT_BASE = ROOT / "data_perp/artifacts/strict_r3_o3v2_market_dynamics_extended_timestamp_labels_20260825_v2/market_dynamics_extended_timestamp_labels.parquet"
DEFAULT_OI = ROOT / "data_perp/exchanges/krakenfutures/open_interest_hourly"
DEFAULT_FUNDING = ROOT / "data_perp/exchanges/krakenfutures/funding_hourly"


def _write_exclusive(path: Path, value: object) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)


def _directory_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(root.glob("*.parquet")):
        digest.update(item.name.encode())
        digest.update(str(item.stat().st_size).encode())
        digest.update(str(item.stat().st_mtime_ns).encode())
    return digest.hexdigest()


def _series(path: Path, field: str, index: pd.DatetimeIndex, *, carry_hours: int) -> np.ndarray | None:
    try:
        try:
            frame = pd.read_parquet(path, columns=["ts", field])
        except Exception:
            frame = pd.read_parquet(path, columns=[field])
        ts = pd.to_datetime(frame.pop("ts"), utc=True, errors="coerce") if "ts" in frame else pd.to_datetime(frame.index, utc=True, errors="coerce")
        value = pd.to_numeric(frame[field], errors="coerce")
        item = pd.Series(value.to_numpy(float), index=ts).loc[lambda s: ~s.index.duplicated(keep="last")]
        aligned = item.reindex(index)
        if carry_hours:
            aligned = aligned.ffill(limit=carry_hours)
        return aligned.to_numpy(np.float32)
    except Exception:
        return None


def _panel(root: Path, field: str, index: pd.DatetimeIndex, *, carry_hours: int) -> tuple[np.ndarray, list[str]]:
    rows: list[np.ndarray] = []
    names: list[str] = []
    paths = sorted(root.glob("*.parquet"))
    # These are independent, immutable source files.  Bounded parallel reads
    # remove the dominant I/O overhead without constructing a full wide
    # dataframe per worker or changing source ordering/provenance.
    reader = partial(_series, field=field, index=index, carry_hours=carry_hours)
    with ThreadPoolExecutor(max_workers=min(16, len(paths))) as executor:
        values = list(executor.map(reader, paths))
    for path, value in zip(paths, values):
        if value is not None:
            rows.append(value)
            names.append(path.stem)
    if not rows:
        raise FileNotFoundError(f"no readable {field} panels in {root}")
    return np.column_stack(rows), names


def _build(oi: np.ndarray, funding: np.ndarray) -> pd.DataFrame:
    n = len(oi)
    output = {
        "market_open_interest_change_12h": np.full(n, np.nan, dtype=np.float32),
        "market_funding_impulse_12h": np.full(n, np.nan, dtype=np.float32),
        "market_liquidation_imbalance_proxy_12h": np.full(n, np.nan, dtype=np.float32),
        "derivatives_label_valid": np.zeros(n, dtype=bool),
    }
    # Align by common first min-assets subset only through aggregate counts;
    # symbols missing on either source simply do not contribute to that label.
    for t in range(n - HOURS):
        oi_now, oi_end = oi[t], oi[t + HOURS]
        oi_mask = np.isfinite(oi_now) & np.isfinite(oi_end) & (oi_now > 0) & (oi_end > 0)
        if int(oi_mask.sum()) >= MIN_OI_ASSETS:
            output["market_open_interest_change_12h"][t] = np.float32(np.log(oi_end[oi_mask].sum() / max(oi_now[oi_mask].sum(), 1e-12)))
        funding_now = funding[t]
        funding_future = funding[t + 1:t + HOURS + 1]
        funding_mask = np.isfinite(funding_now)
        future_means = np.nanmean(funding_future, axis=0)
        valid_funding = funding_mask & np.isfinite(future_means)
        if int(valid_funding.sum()) >= MIN_FUNDING_ASSETS:
            output["market_funding_impulse_12h"][t] = np.float32(np.nanmean(future_means[valid_funding] - funding_now[valid_funding]))
        # Crowding-weighted deleveraging proxy.  A positive baseline funding
        # coupled with OI contraction is long-side deleveraging; a negative
        # baseline funding coupled with the same contraction is short-side
        # deleveraging.  This is deliberately not named an observed
        # liquidation field because the historical source lacks that tape.
        common = oi_mask & np.isfinite(funding_now)
        if int(common.sum()) >= MIN_OI_ASSETS:
            contraction = np.maximum(0.0, np.log(oi_now[common] / oi_end[common]))
            crowding = np.tanh(funding_now[common] * 100_000.0)
            total = contraction.sum()
            if total > 0:
                output["market_liquidation_imbalance_proxy_12h"][t] = np.float32(np.sum(contraction * crowding) / total)
        output["derivatives_label_valid"][t] = all(np.isfinite(output[name][t]) for name in output if name.startswith("market_"))
    return pd.DataFrame(output)


def run(*, base: Path, oi_root: Path, funding_root: Path, out: Path) -> Path:
    if out.exists():
        raise FileExistsError(out)
    base_frame = pd.read_parquet(base)
    base_frame["__decision_ts__"] = pd.to_datetime(base_frame["__decision_ts__"], utc=True, errors="raise")
    if base_frame["__decision_ts__"].duplicated().any():
        raise AssertionError("base label table must have exactly one row per decision timestamp")
    decisions = pd.DatetimeIndex(base_frame["__decision_ts__"])
    hourly = pd.date_range(decisions.min().floor("h"), decisions.max().ceil("h") + pd.Timedelta(hours=HOURS), freq="h", tz="UTC")
    oi, oi_names = _panel(oi_root, "open_interest", hourly, carry_hours=0)
    funding, funding_names = _panel(funding_root, "funding_rate", hourly, carry_hours=HOURS)
    # OI and funding panels differ slightly in coverage.  Map by stable source
    # stem before calling `_build`, preserving only genuinely shared assets.
    common = sorted(set(oi_names) & set(funding_names))
    if len(common) < MIN_OI_ASSETS:
        raise AssertionError(f"insufficient common historical OI/funding assets: {len(common)}")
    oi_idx = [oi_names.index(name) for name in common]
    funding_idx = [funding_names.index(name) for name in common]
    labels = _build(oi[:, oi_idx], funding[:, funding_idx])
    labels["__decision_ts__"] = hourly
    labels["market_label_available_ts"] = labels["__decision_ts__"] + pd.Timedelta(hours=HOURS)
    result = base_frame.merge(labels, on="__decision_ts__", how="left", validate="one_to_one", suffixes=("", "_derivatives"))
    # Existing general market labels retain their own validity.  The new
    # derivative targets have target-specific finite support and availability.
    result = result.drop(columns=[column for column in result if column.endswith("_derivatives")])
    out.mkdir(parents=True, exist_ok=False)
    result.to_parquet(out / "market_dynamics_derivatives_timestamp_labels.parquet", index=False, compression="zstd")
    _write_exclusive(out / "run_manifest.json", {
        "schema": SCHEMA, "scope": "resolved H12 derivative-positioning labels only; prohibited from inference and score receipts",
        "base": str(base.resolve()), "oi_root": str(oi_root.resolve()), "oi_source_hash": _directory_hash(oi_root),
        "funding_root": str(funding_root.resolve()), "funding_source_hash": _directory_hash(funding_root),
        "common_symbols": len(common), "hours": HOURS, "minimum_oi_assets": MIN_OI_ASSETS, "minimum_funding_assets": MIN_FUNDING_ASSETS,
        "availability": "all three target labels become available at decision timestamp + 12h",
        "liquidation_target": "crowding-weighted OI-deleveraging proxy; no observed historical liquidation tape is claimed",
        "coverage": {column: float(result[column].notna().mean()) for column in ["market_open_interest_change_12h", "market_funding_impulse_12h", "market_liquidation_imbalance_proxy_12h"]},
    })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--oi-root", type=Path, default=DEFAULT_OI)
    parser.add_argument("--funding-root", type=Path, default=DEFAULT_FUNDING)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(run(base=args.base.resolve(), oi_root=args.oi_root.resolve(), funding_root=args.funding_root.resolve(), out=args.out.resolve()))


if __name__ == "__main__":
    main()
