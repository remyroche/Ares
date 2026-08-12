#!/usr/bin/env python3
"""Build the causal-admission input for the frozen ATR2 residual stack.

The residual predictions are already strict OOS.  This adapter adds the exact
13-hour label-availability contract used by the stack, a fold-qualified
identity, and validates that every prediction maps to one source ledger row.
It intentionally does not fit a model or change the score.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

LABEL_DELAY = pd.Timedelta(hours=13)


def materialize(prediction_path: Path, ledger_path: Path, out_path: Path) -> Path:
    pred = pd.read_parquet(prediction_path).copy()
    ledger = pd.read_parquet(
        ledger_path,
        columns=["candidate_id", "__ts__", "side_name", "net_bps", "gross_bps"],
    ).copy()
    required_pred = {"candidate_id", "__ts__", "side_name", "net_bps", "gross_bps", "score", "fold"}
    missing = required_pred.difference(pred.columns)
    if missing:
        raise ValueError(f"prediction artifact missing columns: {sorted(missing)}")
    pred["__ts__"] = pd.to_datetime(pred["__ts__"], utc=True, errors="raise")
    ledger["__ts__"] = pd.to_datetime(ledger["__ts__"], utc=True, errors="raise")
    if pred["candidate_id"].duplicated().any():
        raise ValueError("prediction candidate IDs must be unique across transport folds")
    if ledger["candidate_id"].duplicated().any():
        raise ValueError("ledger candidate IDs must be unique")
    src = ledger.set_index("candidate_id")
    ids = pred["candidate_id"]
    missing_ids = ids[~ids.isin(src.index)]
    if len(missing_ids):
        raise ValueError(f"{len(missing_ids)} predictions are absent from the source ledger")
    for col in ("__ts__", "side_name", "net_bps", "gross_bps"):
        left = pred[col].to_numpy()
        right = src.loc[ids, col].to_numpy()
        if col == "__ts__":
            ok = left == right
        elif col in ("net_bps", "gross_bps"):
            ok = pd.Series(left).astype(float).sub(pd.Series(right).astype(float)).abs().to_numpy() <= 0.02
        else:
            ok = left == right
        if not ok.all():
            raise ValueError(f"prediction/ledger mismatch in {col}: {int((~ok).sum())} rows")
    pred["label_available_ts"] = pred["__ts__"] + LABEL_DELAY
    pred["admission_identity"] = pred["fold"].astype(str) + "::" + pred["candidate_id"].astype(str)
    if pred["admission_identity"].duplicated().any():
        raise ValueError("fold-qualified admission identity is not unique")
    pred = pred.sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    out_path.mkdir(parents=True, exist_ok=True)
    target = out_path / "admission_input.parquet"
    pred.to_parquet(target, index=False, compression="zstd")
    manifest = {
        "schema": "frozen_specialist_admission_input_v1",
        "prediction_artifact": str(prediction_path),
        "source_ledger": str(ledger_path),
        "label_delay": str(LABEL_DELAY),
        "label_available_contract": "decision timestamp + 13h; map fitting requires label_available_ts < snapshot decision timestamp",
        "identity": "fold::candidate_id",
        "rows": int(len(pred)),
        "timestamp_min": pred["__ts__"].min().isoformat(),
        "timestamp_max": pred["__ts__"].max().isoformat(),
    }
    (out_path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return target


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--ledger", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    print(materialize(args.predictions, args.ledger, args.out))


if __name__ == "__main__":
    main()
