#!/usr/bin/env python3
"""Combine final-fit historical prediction exports into a ledger-like parquet."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _strategy_prediction_paths(input_dir: Path) -> list[Path]:
    return sorted(input_dir.glob("*/live_finalfit_policy_oos_predictions.parquet"))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument(
        "--require-finite-score",
        action="store_true",
        help="Keep only rows with finite calibrated_score and policy_rank_pct.",
    )
    args = parser.parse_args()

    frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for path in _strategy_prediction_paths(args.input_dir):
        frame = pd.read_parquet(path)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frame["signal_bar_ts"] = frame["timestamp"]
        frame["_source_path"] = str(path)
        before = len(frame)
        if args.require_finite_score:
            finite = np.isfinite(pd.to_numeric(frame["calibrated_score"], errors="coerce"))
            finite &= np.isfinite(pd.to_numeric(frame["policy_rank_pct"], errors="coerce"))
            frame = frame.loc[finite.to_numpy()].copy()
        summaries.append(
            {
                "path": str(path),
                "strategy_id": str(frame["strategy_id"].dropna().iloc[0]) if not frame.empty else "",
                "input_rows": int(before),
                "output_rows": int(len(frame)),
                "timestamp_min": pd.to_datetime(frame["timestamp"], utc=True).min() if not frame.empty else None,
                "timestamp_max": pd.to_datetime(frame["timestamp"], utc=True).max() if not frame.empty else None,
                "timestamps": int(pd.to_datetime(frame["timestamp"], utc=True).nunique()) if not frame.empty else 0,
                "symbols": int(frame["symbol"].astype(str).nunique()) if not frame.empty else 0,
                "finite_base": int(np.isfinite(pd.to_numeric(frame.get("base_pred"), errors="coerce")).sum())
                if not frame.empty
                else 0,
                "finite_meta": int(np.isfinite(pd.to_numeric(frame.get("meta_pred"), errors="coerce")).sum())
                if not frame.empty
                else 0,
            }
        )
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise RuntimeError(f"No prediction rows found under {args.input_dir}")
    out = pd.concat(frames, axis=0, ignore_index=True, copy=False)
    out = out.dropna(subset=["timestamp", "symbol", "strategy_id"])
    out = out.sort_values(["timestamp", "strategy_id", "symbol"], kind="mergesort")
    out = out.drop_duplicates(["timestamp", "strategy_id", "symbol"], keep="last")
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output_path, index=False)
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generated_by": "combine_live_finalfit_predictions_as_ledger",
        "input_dir": str(args.input_dir),
        "output_path": str(args.output_path),
        "require_finite_score": bool(args.require_finite_score),
        "rows": int(len(out)),
        "timestamp_min": pd.to_datetime(out["timestamp"], utc=True).min().isoformat(),
        "timestamp_max": pd.to_datetime(out["timestamp"], utc=True).max().isoformat(),
        "timestamps": int(pd.to_datetime(out["timestamp"], utc=True).nunique()),
        "strategies": summaries,
    }
    manifest_path = args.output_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(_json_safe(manifest), indent=2) + "\n")
    print(json.dumps(_json_safe(manifest), indent=2)[:5000])
    print(f"\nWrote {args.output_path}")


if __name__ == "__main__":
    main()
