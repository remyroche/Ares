#!/usr/bin/env python3
"""Create policy-rank reference sample rows from cached live feature matrices.

The final-fit historical scorer consumes rank-reference parquet files as a
source of ``timestamp``/``symbol`` rows to score.  When the policy optimiser's
original rank references stop before a desired replay interval, cached live
matrices can provide the row universe without changing the trained models.
The placeholder score/rank columns written here are not used as model outputs;
they are retained only to satisfy the existing reference-row schema.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_fixed_tpsl_blend_simple_policy_optimiser import STRATEGY_IDS  # noqa: E402


def _safe_strategy_filename(strategy_id: str) -> str:
    sid = str(strategy_id or "").strip()
    return "".join(ch if ch.isalnum() or ch in "_.=-" else "_" for ch in sid) or "unknown_strategy"


def _matrix_timestamp(path: Path) -> pd.Timestamp | None:
    stem = path.stem
    if not stem.startswith("matrix_"):
        return None
    try:
        return pd.to_datetime(stem[len("matrix_") :], format="%Y%m%dT%H%M%SZ", utc=True)
    except Exception:
        return None


def _matrix_symbols(path: Path) -> list[str]:
    frame = pd.read_parquet(path)
    if frame.empty:
        return []
    if "symbol" in frame.columns:
        values = frame["symbol"].dropna().astype(str).tolist()
    else:
        values = pd.Index(frame.index).dropna().astype(str).tolist()
    return sorted(dict.fromkeys(values))


def _load_matrix_rows(feature_root: Path, start: str, end: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    matrix_root = feature_root / "_live_latest_matrix"
    if not matrix_root.exists():
        raise FileNotFoundError(f"Missing cached matrix directory: {matrix_root}")
    start_ts = pd.Timestamp(start, tz="UTC") if pd.Timestamp(start).tzinfo is None else pd.Timestamp(start).tz_convert("UTC")
    end_ts = pd.Timestamp(end, tz="UTC") if pd.Timestamp(end).tzinfo is None else pd.Timestamp(end).tz_convert("UTC")
    rows: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []
    for path in sorted(matrix_root.glob("matrix_*.parquet")):
        ts = _matrix_timestamp(path)
        if ts is None or ts < start_ts or ts > end_ts:
            continue
        symbols = _matrix_symbols(path)
        manifest_rows.append(
            {
                "timestamp": ts.isoformat(),
                "matrix_path": str(path),
                "symbols": int(len(symbols)),
            }
        )
        if symbols:
            rows.append(pd.DataFrame({"timestamp": ts, "symbol": symbols}))
    if not rows:
        raise RuntimeError(f"No cached matrix rows found in {matrix_root} for {start}..{end}.")
    out = pd.concat(rows, axis=0, ignore_index=True)
    out = out.drop_duplicates(["timestamp", "symbol"]).sort_values(["timestamp", "symbol"], kind="mergesort")
    return out.reset_index(drop=True), manifest_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, default=Path("data_perp/features/20260627_120000"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", default="2026-06-15")
    parser.add_argument("--end", default="2026-06-22 23:59:59")
    parser.add_argument("--strategy-id", action="append", default=[])
    parser.add_argument("--market-mode", default="perps")
    args = parser.parse_args()

    strategy_ids = [str(s).strip() for s in args.strategy_id if str(s).strip()] or list(STRATEGY_IDS.values())
    sample_rows, matrix_manifest = _load_matrix_rows(args.feature_root, args.start, args.end)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    for strategy_id in strategy_ids:
        frame = sample_rows.copy()
        frame.insert(0, "strategy_id", strategy_id)
        frame["calibrated_score"] = 0.5
        frame["rank_pct"] = 0.5
        frame["market_mode"] = str(args.market_mode)
        out_path = args.output_dir / f"{_safe_strategy_filename(strategy_id)}.parquet"
        frame.to_parquet(out_path, index=False)
        summaries.append(
            {
                "strategy_id": strategy_id,
                "path": str(out_path),
                "rows": int(len(frame)),
                "timestamp_min": pd.to_datetime(frame["timestamp"], utc=True).min().isoformat(),
                "timestamp_max": pd.to_datetime(frame["timestamp"], utc=True).max().isoformat(),
                "timestamps": int(pd.to_datetime(frame["timestamp"], utc=True).nunique()),
                "symbols": int(frame["symbol"].astype(str).nunique()),
            }
        )
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generated_by": "create_rank_reference_from_live_matrices",
        "feature_root": str(args.feature_root),
        "output_dir": str(args.output_dir),
        "start": args.start,
        "end": args.end,
        "matrix_count": int(len(matrix_manifest)),
        "matrix_rows": matrix_manifest,
        "strategies": summaries,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2)[:5000])
    print(f"\nWrote {args.output_dir}")


if __name__ == "__main__":
    main()
