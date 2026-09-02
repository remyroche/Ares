#!/usr/bin/env python3
"""Add exact, training-only economic target variants to short R3 labels.

The input is the immutable side-local exact-H12 label substrate.  This
materializer reopens only the post-decision one-minute path in order to derive
cost-aware robust-clear timing and same-bar ambiguity.  It never changes a
candidate, causal feature, entry, ATR, or label-validity decision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_packb_tp6_sl4_h12_labels import (  # noqa: E402
    _minute_path_pruned,
    _packb_to_kraken_symbol,
)


HORIZON_MINUTES = 12 * 60
COST_BPS = 100.0
MEANINGFUL_ADVERSE_ATR = 4.0
SOFT_TEMPERATURE_BPS = 50.0
TIME_SCALE_MINUTES = 4.0 * 60.0
TIME_FLOOR = 0.60


@njit(cache=True)
def _derive_robust_clear(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    starts: np.ndarray,
    entry: np.ndarray,
    atr: np.ndarray,
    buffers_bps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return complete, pre-adverse MFE, clear time, lower time and ambiguity.

    The input is short-only.  A simultaneous favourable threshold and
    meaningful adverse touch belongs to the adverse path, matching the frozen
    strict-R3 same-minute convention.
    """
    rows = len(starts)
    n_buffers = len(buffers_bps)
    complete = np.zeros(rows, dtype=np.bool_)
    pre_adverse_mfe_bps = np.empty(rows, dtype=np.float32)
    pre_adverse_mfe_bps[:] = np.nan
    first_clear = np.full((rows, n_buffers), -1, dtype=np.int16)
    lower_touch = np.full(rows, -1, dtype=np.int16)
    ambiguity = np.zeros((rows, n_buffers), dtype=np.bool_)
    for row in range(rows):
        start = starts[row]
        e = entry[row]
        a = atr[row]
        if start < 0 or start + HORIZON_MINUTES > len(close) or not np.isfinite(e) or not np.isfinite(a) or e <= 0.0 or a <= 0.0:
            continue
        best_bps = 0.0
        valid = True
        atr_bps = a / e * 10000.0
        for offset in range(HORIZON_MINUTES):
            pos = start + offset
            h = high[pos]
            l = low[pos]
            c = close[pos]
            if not np.isfinite(h) or not np.isfinite(l) or not np.isfinite(c):
                valid = False
                break
            favourable_bps = (e - l) / e * 10000.0
            adverse_atr = (h - e) / a
            # Same-minute lower/adverse priority: before recording a first
            # robust clear, retain an explicit ambiguity receipt.
            if adverse_atr >= MEANINGFUL_ADVERSE_ATR:
                lower_touch[row] = offset + 1
                for column in range(n_buffers):
                    if first_clear[row, column] < 0 and favourable_bps > COST_BPS + buffers_bps[column]:
                        ambiguity[row, column] = True
                break
            if favourable_bps > best_bps:
                best_bps = favourable_bps
            for column in range(n_buffers):
                if first_clear[row, column] < 0 and favourable_bps > COST_BPS + buffers_bps[column]:
                    first_clear[row, column] = offset + 1
        if valid:
            complete[row] = True
            pre_adverse_mfe_bps[row] = best_bps
    return complete, pre_adverse_mfe_bps, first_clear, lower_touch, ambiguity


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -35.0, 35.0)))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _month_input(root: Path, month: pd.Timestamp) -> Path:
    path = root / "parts" / f"month={month:%Y-%m}" / "side=short.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _augment_month(part: pd.DataFrame, minute_root: Path, buffers: tuple[int, ...]) -> pd.DataFrame:
    out = part.copy()
    valid = (
        out["label_valid"].astype("boolean").fillna(False).astype(bool)
        & ~out["target_invalid"].astype("boolean").fillna(True).astype(bool)
    )
    for buffer in buffers:
        prefix = f"r3_b{buffer}"
        out[f"{prefix}_margin_bps"] = np.nan
        out[f"{prefix}_first_clear_minute"] = np.int16(-1)
        out[f"{prefix}_same_bar_ambiguity"] = False
        out[f"{prefix}_robust_clear"] = np.nan
        out[f"{prefix}_adverse_first"] = np.nan
        out[f"{prefix}_weak"] = np.nan
        out[f"{prefix}_soft_clear"] = np.nan
        out[f"{prefix}_soft_adverse"] = np.nan
        out[f"{prefix}_soft_weak"] = np.nan
    out["r3_meaningful_adverse_first_minute"] = np.int16(-1)
    for symbol, subset in out.loc[valid].groupby("__symbol__", sort=True):
        indices = subset.index.to_numpy()
        decision = pd.to_datetime(subset["__decision_ts__"], utc=True)
        minute = _minute_path_pruned(
            minute_root,
            _packb_to_kraken_symbol(str(symbol)),
            decision.min(),
            decision.max() + pd.Timedelta(minutes=HORIZON_MINUTES),
        )
        starts = minute.index.get_indexer(decision).astype(np.int64)
        complete, mfe_bps, clear, lower, ambiguity = _derive_robust_clear(
            minute.high.to_numpy(dtype=np.float64),
            minute.low.to_numpy(dtype=np.float64),
            minute.close.to_numpy(dtype=np.float64),
            starts,
            pd.to_numeric(subset["tp6_sl4_entry_price"], errors="coerce").to_numpy(dtype=np.float64),
            pd.to_numeric(subset["atr_1h"], errors="coerce").to_numpy(dtype=np.float64),
            np.asarray(buffers, dtype=np.float64),
        )
        if not complete.all():
            raise AssertionError(
                f"exact target augmentation lost a previously complete short path: {symbol}"
            )
        # The existing canonical label is the same pre-adverse construction at
        # B25.  Verify the reopened direction-specific path rather than
        # silently accepting a timestamp or price shift.
        existing = pd.to_numeric(subset["pre_adverse_mfe_bps"], errors="coerce").to_numpy(float)
        if not np.allclose(mfe_bps, existing, rtol=0.0, atol=2e-3, equal_nan=False):
            raise AssertionError(f"pre-adverse MFE differs from frozen short label: {symbol}")
        out.loc[indices, "r3_meaningful_adverse_first_minute"] = lower
        for column, buffer in enumerate(buffers):
            prefix = f"r3_b{buffer}"
            first = clear[:, column]
            adverse = first < 0
            adverse &= lower >= 0
            robust = first >= 0
            weak = ~(robust | adverse)
            margin = mfe_bps.astype(np.float64) - COST_BPS - float(buffer)
            time_factor = np.zeros(len(first), dtype=float)
            time_factor[robust] = (
                TIME_FLOOR
                + (1.0 - TIME_FLOOR)
                * np.exp(-first[robust].astype(float) / TIME_SCALE_MINUTES)
            )
            soft_clear = _sigmoid(margin / SOFT_TEMPERATURE_BPS) * time_factor
            soft_clear[adverse] = 0.0
            soft_adverse = adverse.astype(float)
            soft_weak = 1.0 - soft_clear - soft_adverse
            if np.any(soft_weak < -1e-7):
                raise AssertionError("soft R3 memberships leave the probability simplex")
            out.loc[indices, f"{prefix}_margin_bps"] = margin.astype(np.float32)
            out.loc[indices, f"{prefix}_first_clear_minute"] = first
            out.loc[indices, f"{prefix}_same_bar_ambiguity"] = ambiguity[:, column]
            out.loc[indices, f"{prefix}_robust_clear"] = robust.astype(np.float32)
            out.loc[indices, f"{prefix}_adverse_first"] = adverse.astype(np.float32)
            out.loc[indices, f"{prefix}_weak"] = weak.astype(np.float32)
            out.loc[indices, f"{prefix}_soft_clear"] = soft_clear.astype(np.float32)
            out.loc[indices, f"{prefix}_soft_adverse"] = soft_adverse.astype(np.float32)
            out.loc[indices, f"{prefix}_soft_weak"] = soft_weak.astype(np.float32)
    return out


def run(
    *, labels_root: Path, minute_root: Path, out: Path,
    start: pd.Timestamp, end: pd.Timestamp, buffers: tuple[int, ...],
) -> Path:
    if out.exists():
        raise FileExistsError(f"output must be new: {out}")
    if not buffers or any(buffer <= 0 for buffer in buffers):
        raise ValueError("buffers must be positive bps")
    records: list[dict[str, object]] = []
    for month in pd.date_range(start, end, freq="MS", inclusive="left"):
        source = _month_input(labels_root, month)
        part = pd.read_parquet(source)
        if not part.side_name.astype(str).str.lower().eq("short").all():
            raise ValueError(f"input labels are not short-only: {source}")
        augmented = _augment_month(part, minute_root, buffers)
        path = out / "parts" / f"month={month:%Y-%m}" / "side=short.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        augmented.to_parquet(path, index=False, compression="zstd")
        valid = augmented.label_valid.astype(bool)
        record: dict[str, object] = {
            "month": f"{month:%Y-%m}", "rows": int(len(augmented)),
            "valid_rows": int(valid.sum()), "invalid_rows": int((~valid).sum()),
        }
        for buffer in buffers:
            prefix = f"r3_b{buffer}"
            record[f"clear_rate_b{buffer}"] = float(augmented.loc[valid, f"{prefix}_robust_clear"].mean())
            record[f"adverse_rate_b{buffer}"] = float(augmented.loc[valid, f"{prefix}_adverse_first"].mean())
            record[f"ambiguity_rate_b{buffer}"] = float(augmented.loc[valid, f"{prefix}_same_bar_ambiguity"].mean())
        records.append(record)
        print(json.dumps(record), flush=True)
    pd.DataFrame(records).to_parquet(out / "coverage.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "strict_r3_short_exact_target_augmentation_v1",
        "status": "complete",
        "side": "short",
        "input_labels": str(labels_root),
        "input_manifest_sha256": _sha256(labels_root / "run_manifest.json"),
        "minute_root": str(minute_root),
        "months": [record["month"] for record in records],
        "entry": "frozen exact one-minute open at signal close + one hour",
        "horizon": "12 hours",
        "meaningful_adverse": "short adverse move >= 4 ATR; same-minute tie is adverse-first",
        "cost_bps": COST_BPS,
        "robust_clear_buffers_bps": list(buffers),
        "soft_r3": {
            "margin_temperature_bps": SOFT_TEMPERATURE_BPS,
            "time_scale_minutes": TIME_SCALE_MINUTES,
            "time_floor": TIME_FLOOR,
            "memberships": "robust-clear / adverse-first / weak; normalized on every valid row",
        },
        "future_data": "used only for persisted supervised labels; no augmented column is a model input",
        "coverage": records,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--minute-root", type=Path, default=ROOT / "data_perp/exchanges/krakenfutures/execution_1m/ohlcv")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start", default="2024-01-01T00:00:00Z")
    parser.add_argument("--end", default="2024-07-01T00:00:00Z")
    parser.add_argument("--buffers-bps", default="50,75")
    args = parser.parse_args()
    start = pd.to_datetime(args.start, utc=True)
    end = pd.to_datetime(args.end, utc=True)
    if start.day != 1 or start.hour != 0 or end.day != 1 or end.hour != 0:
        raise ValueError("target augmentation requires UTC month boundaries")
    buffers = tuple(sorted({int(value) for value in args.buffers_bps.split(",") if value.strip()}))
    print(run(labels_root=args.labels_root, minute_root=args.minute_root, out=args.out, start=start, end=end, buffers=buffers))


if __name__ == "__main__":
    main()
