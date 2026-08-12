"""Canonicalize historical Kraken PF OI/funding sidecars.

Historical exports do not carry a reliable ingestion timestamp.  The approved
research contract therefore declares each observation available at the next
hour, shifts the sidecar index by one hour, and leaves bounded freshness to
the feature loader.  This is an explicit assumption, not evidence of live
availability; the manifest records it so production promotion can require a
future parity audit.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def _clean(path: Path, value_columns: tuple[str, ...], *, write: bool = True) -> dict[str, object]:
    frame = pd.read_parquet(path)
    if frame.empty:
        return {"rows_in": 0, "rows_out": 0, "dropped": 0}
    index = pd.to_datetime(frame.index, utc=True, errors="coerce").floor("h")
    keep = ~index.isna()
    frame = frame.loc[keep].copy()
    index = index[keep]
    value = next((c for c in value_columns if c in frame.columns), None)
    if value is None:
        return {"rows_in": int(len(frame)), "rows_out": 0, "dropped": int(len(frame))}
    values = pd.to_numeric(frame[value], errors="coerce").to_numpy(float)
    keep = np.isfinite(values) & (values > 0.0)
    observed = pd.DataFrame(
        {value: values[keep].astype(np.float32)},
        index=pd.DatetimeIndex(index[keep], name="observation_ts"),
    )
    observed = observed[~observed.index.duplicated(keep="last")].sort_index()
    # Approved historical availability assumption: next hourly bar.
    available = observed.copy()
    available.index = pd.DatetimeIndex(
        available.index + pd.Timedelta(hours=1), name="ts"
    )
    if write:
        temporary = path.with_name(f".{path.name}.availability.tmp")
        available.to_parquet(temporary, engine="pyarrow", compression="zstd")
        os.replace(temporary, path)
    return {
        "rows_in": int(len(frame)),
        "rows_out": int(len(available)),
        "dropped": int(len(frame) - len(available)),
        "first_available_ts": available.index.min().isoformat() if len(available) else None,
        "last_available_ts": available.index.max().isoformat() if len(available) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data_perp/exchanges/krakenfutures"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--resume-mtime-cutoff",
        type=str,
        default=None,
        help=(
            "Resume a known interrupted run: files modified at or after this UTC "
            "timestamp are already availability-indexed and are not shifted again. "
            "Files before the cutoff are canonicalized once."
        ),
    )
    args = parser.parse_args()
    cutoff = pd.Timestamp(args.resume_mtime_cutoff, tz="UTC") if args.resume_mtime_cutoff else None
    result: dict[str, object] = {
        "schema": "kraken_pf_oi_funding_sidecar_v2",
        "status": "COMPLETED_RESEARCH_CONTRACT",
        "product_filter": "canonical *_USD_USD files only (PF linear USD convention)",
        "availability_rule": "availability_ts = observation_ts + 1h",
        "availability_rule_assumption": True,
        "forward_fill": "none in sidecar; bounded carry remains in feature loader",
        "families": {},
    }
    for family, value_columns in (
        ("open_interest_hourly", ("open_interest", "openInterest", "openInterestValue")),
        ("funding_hourly", ("funding_rate", "relativeFundingRate", "fundingRate")),
    ):
        directory = args.root / family
        files = sorted(directory.glob("*_USD_USD.parquet"))
        rows = {}
        for path in files:
            already_canonical = cutoff is not None and path.stat().st_mtime >= cutoff.timestamp()
            rows[path.name] = {
                **_clean(path, value_columns, write=not already_canonical),
                "already_canonical_from_interrupted_run": bool(already_canonical),
            }
        result["families"][family] = {"files": len(files), "rows": rows}
    result["resume_mtime_cutoff_utc"] = cutoff.isoformat() if cutoff is not None else None
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
