#!/usr/bin/env python3
"""Add selected fold-cache feature columns back to a scored base ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


def _feature_columns(feature_selection_path: Path, pattern: str) -> list[str]:
    selection = pd.read_csv(feature_selection_path)
    if "selected" in selection.columns:
        selection = selection[selection["selected"].fillna(False).astype(bool)]
    mask = selection["feature"].astype(str).str.contains(pattern, case=False, regex=True, na=False)
    cols = selection.loc[mask, "feature"].astype(str).drop_duplicates().tolist()
    return cols


def _side_name(frame: pd.DataFrame) -> pd.Series:
    if "side_name" in frame.columns:
        return frame["side_name"].astype(str).str.lower()
    side = pd.to_numeric(frame.get("__side__", frame.get("side", 1.0)), errors="coerce").fillna(1.0)
    return side.lt(0.0).map({True: "short", False: "long"})


def run(args: argparse.Namespace) -> dict[str, Any]:
    ledger_path = Path(args.ledger)
    fold_cache_dir = Path(args.fold_cache_dir)
    feature_selection_path = Path(args.feature_selection)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    features = _feature_columns(feature_selection_path, str(args.feature_pattern))
    if not features:
        raise RuntimeError(f"No selected features matched pattern: {args.feature_pattern}")

    ledger = pd.read_parquet(ledger_path)
    ledger["__ts__"] = pd.to_datetime(ledger["__ts__"], utc=True, errors="coerce").dt.tz_convert(None)
    ledger["side_name"] = _side_name(ledger)
    key_cols = ["oos_fold", "__ts__", "__symbol__", "side_name"]

    parts: list[pd.DataFrame] = []
    missing_folds: list[str] = []
    used_features: set[str] = set()
    for fold in sorted(ledger["oos_fold"].dropna().astype(str).unique()):
        fold_dir = fold_cache_dir / fold
        valid_path = fold_dir / "valid.parquet"
        x_valid_path = fold_dir / "x_valid.parquet"
        if not valid_path.exists() or not x_valid_path.exists():
            missing_folds.append(fold)
            continue
        valid = pd.read_parquet(valid_path)
        valid["__ts__"] = pd.to_datetime(valid["__ts__"], utc=True, errors="coerce").dt.tz_convert(None)
        valid["side_name"] = _side_name(valid)
        valid["oos_fold"] = fold
        x_cols = set(pq.read_schema(x_valid_path).names)
        read_cols = [col for col in features if col in x_cols]
        if not read_cols:
            continue
        x_valid = pd.read_parquet(x_valid_path, columns=read_cols)
        used_features.update(read_cols)
        part = pd.concat(
            [
                valid.loc[:, ["oos_fold", "__ts__", "__symbol__", "side_name"]].reset_index(drop=True),
                x_valid.loc[:, read_cols].reset_index(drop=True),
            ],
            axis=1,
        )
        parts.append(part)

    if not parts:
        raise RuntimeError("No fold-cache feature parts were built")

    feature_frame = pd.concat(parts, ignore_index=True)
    feature_frame = feature_frame.drop_duplicates(key_cols, keep="last")
    before_cols = set(ledger.columns)
    out = ledger.merge(feature_frame, on=key_cols, how="left", validate="one_to_one")
    added_cols = [col for col in sorted(used_features) if col in out.columns and col not in before_cols]
    matched = out[added_cols].notna().any(axis=1) if added_cols else pd.Series(False, index=out.index)
    out.to_parquet(output_path, index=False)
    manifest = {
        "ledger": str(ledger_path),
        "fold_cache_dir": str(fold_cache_dir),
        "feature_selection": str(feature_selection_path),
        "output": str(output_path),
        "feature_pattern": str(args.feature_pattern),
        "requested_feature_count": int(len(features)),
        "added_feature_count": int(len(added_cols)),
        "added_features": added_cols,
        "rows": int(len(out)),
        "matched_rows": int(matched.sum()),
        "match_rate": float(matched.mean()) if len(out) else 0.0,
        "missing_folds": missing_folds,
    }
    Path(str(output_path) + ".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--fold-cache-dir", required=True, type=Path)
    parser.add_argument("--feature-selection", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--feature-pattern",
        default="gmm|mahalan|reconstruction|cluster|latent|posterior",
        help="Regex applied to selected feature names before reading fold-cache x_valid files.",
    )
    return parser.parse_args()


def main() -> int:
    manifest = run(parse_args())
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
