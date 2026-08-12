"""Materialize causal OI/funding freshness and eligibility masks.

Masks are derived only from availability-indexed sidecars and the feature
timestamp.  They are metadata columns, not economic model inputs.  Funding
has two deliberately separate contracts: the raw funding alias uses the
strict two-bar carry in ``features.py``, while event-derived funding features
allow the bounded 24-hour event freshness used by ``features_oi.py``.  Keeping
both masks prevents a broad event-eligibility flag from silently authorising
the raw alias.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backfill_kraken_oi_funding_sidecars import _parse_pf_product, _safe_key


OI_CARRY_BARS = 3
FUNDING_ALIAS_CARRY_BARS = 2
FUNDING_EVENT_CARRY_BARS = 24


def _idx(path: Path) -> pd.DatetimeIndex:
    if not path.exists():
        return pd.DatetimeIndex([], tz="UTC")
    frame = pd.read_parquet(path, columns=[])
    idx = pd.to_datetime(frame.index, utc=True, errors="coerce")
    return pd.DatetimeIndex(idx[~idx.isna()]).drop_duplicates().sort_values()


def _fresh_bars(feature_idx: pd.DatetimeIndex, source_idx: pd.DatetimeIndex) -> np.ndarray:
    """Return causal hourly age; -1 means no prior source observation."""
    out = np.full(len(feature_idx), -1, dtype=np.int16)
    if not len(feature_idx) or not len(source_idx):
        return out
    source_ns = source_idx.view("i8")
    feature_ns = feature_idx.view("i8")
    pos = np.searchsorted(source_ns, feature_ns, side="right") - 1
    valid = pos >= 0
    if valid.any():
        age = ((feature_ns[valid] - source_ns[pos[valid]]) // int(pd.Timedelta(hours=1).value)).astype(np.int64)
        out[valid] = np.clip(age, -1, np.iinfo(np.int16).max).astype(np.int16)
    return out


def _atomic_write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False) as handle:
        tmp = Path(handle.name)
    try:
        frame.to_parquet(tmp, engine="pyarrow", compression="zstd")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-dir", type=Path, required=True)
    ap.add_argument("--perp-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--exclude-finalization-json", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    excluded: set[str] = set()
    if args.exclude_finalization_json and args.exclude_finalization_json.exists():
        excluded = set(map(str, json.loads(args.exclude_finalization_json.read_text()).get("stale_files_cleared", [])))
    results: list[dict[str, Any]] = []
    for path in sorted(args.feature_dir.glob("symbol=*.parquet")):
        symbol = path.stem.removeprefix("symbol=")
        if symbol in excluded or _parse_pf_product(symbol).status != "READY":
            continue
        frame = pd.read_parquet(path)
        if "ts" in frame.columns:
            frame["ts"] = pd.to_datetime(frame["ts"], utc=True, errors="coerce")
            frame = frame.set_index("ts")
        else:
            frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce")
        frame = frame.loc[~frame.index.isna()].sort_index()
        feature_idx = pd.DatetimeIndex(frame.index)
        key = _safe_key(symbol)
        oi_idx = _idx(args.perp_root / "open_interest_hourly" / f"{key}.parquet")
        funding_idx = _idx(args.perp_root / "funding_hourly" / f"{key}.parquet")
        oi_age = _fresh_bars(feature_idx, oi_idx)
        funding_age = _fresh_bars(feature_idx, funding_idx)
        oi_exact = np.asarray(feature_idx.isin(oi_idx), dtype=bool)
        funding_exact = np.asarray(feature_idx.isin(funding_idx), dtype=bool)
        frame["__oi_fresh_bars__"] = oi_age
        frame["__funding_fresh_bars__"] = funding_age
        frame["__oi_exact_available__"] = oi_exact
        frame["__funding_exact_available__"] = funding_exact
        frame["__oi_available__"] = (oi_age >= 0) & (oi_age <= OI_CARRY_BARS)
        frame["__funding_alias_available__"] = (funding_age >= 0) & (funding_age <= FUNDING_ALIAS_CARRY_BARS)
        frame["__funding_available__"] = (funding_age >= 0) & (funding_age <= FUNDING_EVENT_CARRY_BARS)
        if not args.dry_run:
            _atomic_write(frame, path)
        results.append({
            "feature_symbol": symbol,
            "rows": int(len(frame)),
            "oi_exact_rows": int(oi_exact.sum()),
            "oi_eligible_rows": int(frame["__oi_available__"].sum()),
            "funding_exact_rows": int(funding_exact.sum()),
            "funding_alias_eligible_rows": int(frame["__funding_alias_available__"].sum()),
            "funding_eligible_rows": int(frame["__funding_available__"].sum()),
        })
    args.out_dir.mkdir(parents=True, exist_ok=True)
    audit = pd.DataFrame(results)
    audit.to_parquet(args.out_dir / "oi_funding_eligibility_by_asset.parquet", index=False, compression="zstd")
    payload = {
        "schema": "oi_funding_causal_eligibility_masks_v1",
        "status": "DRY_RUN" if args.dry_run else "COMPLETE",
        "feature_dir": str(args.feature_dir),
        "symbols": len(results),
        "oi_carry_bars": OI_CARRY_BARS,
        "funding_alias_carry_bars": FUNDING_ALIAS_CARRY_BARS,
        "funding_event_carry_bars": FUNDING_EVENT_CARRY_BARS,
        "columns": ["__oi_fresh_bars__", "__funding_fresh_bars__", "__oi_exact_available__", "__funding_exact_available__", "__oi_available__", "__funding_alias_available__", "__funding_available__"],
        "contract": "masks use only prior availability-indexed sidecar timestamps; no future fill; __funding_alias_available__ matches the raw funding alias and __funding_available__ matches event-derived funding features",
        "artifacts": ["oi_funding_eligibility_by_asset.parquet"],
    }
    (args.out_dir / "eligibility_mask_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "symbols": len(results), "oi_eligible_rows": int(audit.oi_eligible_rows.sum()) if len(audit) else 0, "funding_eligible_rows": int(audit.funding_eligible_rows.sum()) if len(audit) else 0}, sort_keys=True))


if __name__ == "__main__":
    main()
