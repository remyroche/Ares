"""Verify causal OI/funding masks and raw funding alias parity.

This is a lightweight inference canary.  It reads only metadata plus the raw
funding alias from each feature file, recomputes source ages from the
availability-indexed sidecars, and fails closed on any mismatch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from backfill_kraken_oi_funding_sidecars import _parse_pf_product, _safe_key
from finalize_oi_funding_feature_store import SKIPPED_KEYS


def _idx(path: Path) -> pd.DatetimeIndex:
    if not path.exists():
        return pd.DatetimeIndex([], tz="UTC")
    frame = pd.read_parquet(path, columns=[])
    idx = pd.to_datetime(frame.index, utc=True, errors="coerce")
    return pd.DatetimeIndex(idx[~idx.isna()]).drop_duplicates().sort_values()


def _age(feature_idx: pd.DatetimeIndex, source_idx: pd.DatetimeIndex) -> np.ndarray:
    out = np.full(len(feature_idx), -1, dtype=np.int64)
    if not len(feature_idx) or not len(source_idx):
        return out
    source_ns = source_idx.view("i8")
    feature_ns = feature_idx.view("i8")
    pos = np.searchsorted(source_ns, feature_ns, side="right") - 1
    valid = pos >= 0
    if valid.any():
        out[valid] = (feature_ns[valid] - source_ns[pos[valid]]) // int(pd.Timedelta(hours=1).value)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-dir", type=Path, required=True)
    ap.add_argument("--perp-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--exclude-finalization-json", type=Path)
    args = ap.parse_args()

    excluded: set[str] = set()
    if args.exclude_finalization_json and args.exclude_finalization_json.exists():
        excluded = set(json.loads(args.exclude_finalization_json.read_text()).get("stale_files_cleared", []))

    errors: list[dict[str, object]] = []
    checked = 0
    mask_rows = 0
    for path in sorted(args.feature_dir.glob("symbol=*.parquet")):
        symbol = path.stem.removeprefix("symbol=")
        if symbol in excluded or _parse_pf_product(symbol).status != "READY":
            continue
        required_columns = [
            "__oi_fresh_bars__",
            "__funding_fresh_bars__",
            "__oi_exact_available__",
            "__funding_exact_available__",
            "__oi_available__",
            "__funding_alias_available__",
            "__funding_available__",
        ]
        available_columns = set(pq.read_schema(path).names)
        missing = [c for c in required_columns if c not in available_columns]
        if missing:
            errors.append({"symbol": symbol, "error": "missing_columns", "columns": missing})
            continue
        frame = pd.read_parquet(path, columns=required_columns)
        idx = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True, errors="coerce"))
        if idx.isna().any() or idx.has_duplicates or not idx.is_monotonic_increasing:
            errors.append({"symbol": symbol, "error": "invalid_feature_index"})
            continue
        key = _safe_key(symbol)
        oi_idx = _idx(args.perp_root / "open_interest_hourly" / f"{key}.parquet")
        funding_idx = _idx(args.perp_root / "funding_hourly" / f"{key}.parquet")
        oi_age = _age(idx, oi_idx)
        funding_age = _age(idx, funding_idx)
        expected = {
            "__oi_fresh_bars__": oi_age,
            "__funding_fresh_bars__": funding_age,
            "__oi_exact_available__": np.asarray(idx.isin(oi_idx), dtype=bool),
            "__funding_exact_available__": np.asarray(idx.isin(funding_idx), dtype=bool),
            "__oi_available__": (oi_age >= 0) & (oi_age <= 3),
            "__funding_alias_available__": (funding_age >= 0) & (funding_age <= 2),
            "__funding_available__": (funding_age >= 0) & (funding_age <= 24),
        }
        for col, expected_values in expected.items():
            got = frame[col].to_numpy()
            if not np.array_equal(got, expected_values):
                errors.append({"symbol": symbol, "error": "mask_mismatch", "column": col})
        if "funding_rate" not in available_columns and len(funding_idx):
            errors.append({"symbol": symbol, "error": "funding_rate_missing_with_source"})
        if "funding_rate" in available_columns:
            source = pd.to_numeric(pd.read_parquet(path, columns=["funding_rate"])["funding_rate"], errors="coerce")
            expected_rate = pd.Series(np.nan, index=idx, dtype="float32")
        else:
            source = pd.Series(np.nan, index=idx, dtype="float32")
            expected_rate = pd.Series(np.nan, index=idx, dtype="float32")
        if len(funding_idx):
            raw = pd.read_parquet(args.perp_root / "funding_hourly" / f"{key}.parquet")
            raw.index = pd.to_datetime(raw.index, utc=True, errors="coerce")
            value_col = "funding_rate" if "funding_rate" in raw.columns else raw.columns[0]
            expected_rate = pd.to_numeric(raw[value_col], errors="coerce").reindex(idx).ffill(limit=2)
        if "funding_rate" in available_columns and not np.allclose(source.to_numpy(dtype=float), expected_rate.to_numpy(dtype=float), equal_nan=True, atol=1e-6, rtol=1e-6):
            errors.append({"symbol": symbol, "error": "funding_rate_alias_mismatch"})
        checked += 1
        mask_rows += len(frame)

    payload = {
        "schema": "oi_funding_feature_store_parity_v1",
        "status": "PASS" if not errors else "FAIL",
        "checked_symbols": checked,
        "checked_rows": mask_rows,
        "errors": errors[:100],
        "error_count": len(errors),
        "unsupported_keys_fail_closed": sorted(SKIPPED_KEYS),
        "contract": {
            "sidecar_availability": "observation_ts + 1h",
            "feature_to_decision": "feature bar t enters at next candle boundary t+1h",
            "oi_carry_bars": 3,
            "funding_alias_carry_bars": 2,
            "funding_event_carry_bars": 24,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: payload[k] for k in ("status", "checked_symbols", "checked_rows", "error_count")}, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
