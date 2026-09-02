#!/usr/bin/env python3
"""Audit an immutable P8U one-row executor checkpoint against frozen features.

The audit is intentionally target-free: it truncates only the saved source
panel at the requested source timestamp, creates an isolated executor state,
and compares its currently direct sealed outputs with a persisted canonical
feature checkpoint.  It cannot invoke the broad batch graph or emit scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.inference.p8u_single_timestamp_graph import (  # noqa: E402
    P8UOneTimestampExecutor,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(value: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-state", type=Path, required=True)
    parser.add_argument("--reference-features", type=Path, required=True)
    parser.add_argument("--source-timestamp", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.out_dir.exists():
        raise FileExistsError(f"immutable audit output exists: {args.out_dir}")
    source_path = args.source_state.resolve()
    reference_path = args.reference_features.resolve()
    if ROOT not in source_path.parents or ROOT not in reference_path.parents or ROOT not in args.out_dir.resolve().parents:
        raise ValueError("all P8U audit paths must remain below repository root")
    source = joblib.load(source_path)
    if not isinstance(source, dict) or not isinstance(source.get("panel"), dict):
        raise ValueError("source state lacks target-free panel")
    symbols = tuple(map(str, source.get("symbols", ())))
    if len(symbols) != 160:
        raise ValueError("source state lacks frozen 160-symbol universe")
    stamp = _utc(args.source_timestamp)
    panel = {name: frame.loc[:stamp] for name, frame in source["panel"].items()}
    if not panel or any(not isinstance(frame, pd.DataFrame) for frame in panel.values()):
        raise ValueError("source state panels are invalid")
    if any(pd.DatetimeIndex(frame.index).max() != stamp for frame in panel.values()):
        raise ValueError("requested source timestamp is unavailable in every source panel")
    reference = pd.read_parquet(reference_path)
    if "candidate_id" not in reference or "__decision_ts__" not in reference:
        raise ValueError("reference lacks frozen candidate identities")
    source_keys = reference["candidate_id"].astype(str).str.rsplit("|", n=2).str[0]
    source_candidates = reference.assign(__source_symbol__=source_keys)
    if not (source_candidates["__decision_ts__"] == stamp + pd.Timedelta(hours=1)).all():
        raise ValueError("reference decision timestamps do not match source timestamp plus one hour")

    args.out_dir.mkdir(parents=True)
    state_root = args.out_dir / "isolated_direct_state"
    executor = P8UOneTimestampExecutor(root=state_root, symbols=symbols, market_basket=symbols)
    try:
        output = executor.bootstrap(panel)
    except Exception:
        shutil.rmtree(args.out_dir, ignore_errors=True)
        raise
    available = sorted(set(output).intersection(reference.columns))
    rows: list[dict[str, object]] = []
    for name in available:
        direct = pd.Series(output[name], index=symbols)
        paired = source_candidates[["__source_symbol__", name]].assign(
            __direct__=lambda frame: frame["__source_symbol__"].map(direct)
        ).dropna()
        delta = (paired[name].astype(np.float64) - paired["__direct__"].astype(np.float64)).abs()
        rows.append(
            {
                "feature": name,
                "overlap": int(len(paired)),
                "max_abs_delta": float(delta.max()) if len(delta) else None,
                "mean_abs_delta": float(delta.mean()) if len(delta) else None,
            }
        )
    feature_audit = pd.DataFrame(rows).sort_values("feature")
    feature_audit.to_parquet(args.out_dir / "direct_feature_parity.parquet", index=False)
    receipt = {
        "schema": "strict_r3_p8u_direct_reference_parity_v1",
        "status": "pass_target_free_direct_feature_parity",
        "source_timestamp": stamp.isoformat(),
        "decision_timestamp": (stamp + pd.Timedelta(hours=1)).isoformat(),
        "source_state": str(source_path),
        "source_state_sha256": _sha256(source_path),
        "reference_features": str(reference_path),
        "reference_features_sha256": _sha256(reference_path),
        "symbols": len(symbols),
        "direct_feature_count": len(output),
        "sealed_reference_overlap": int(len(feature_audit)),
        "max_abs_delta": float(feature_audit["max_abs_delta"].max()) if len(feature_audit) else None,
        "outcome_columns_consumed": [],
        "model_scores_emitted": False,
        "executor_contract_hash": executor.contract_hash,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
