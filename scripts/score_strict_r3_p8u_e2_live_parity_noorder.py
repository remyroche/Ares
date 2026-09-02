#!/usr/bin/env python3
"""Create one immutable no-order E2 selection from a staged P8U score commit.

The runner expects the caller to supply a current target-free 15-minute E2
feature panel produced before any outcome join.  It cannot open orders and is
intended as the parity boundary that must be validated before an exchange
gateway is separately resealed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.inference.p8u_e2_h4_live_parity import (
    P8UE2H4LiveParityBundle,
    apply_e2_replacement,
)
from extreme_price_movements.p8u_15m_features import FIFTEEN_MINUTE_FEATURE_KEYS, VWAP_15M_FEATURE_KEYS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--staged-commit", type=Path, required=True)
    parser.add_argument("--e2-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("E2 no-order output must be immutable")
    bundle = P8UE2H4LiveParityBundle.load(args.bundle.resolve())
    commit = args.staged_commit.resolve()
    receipt_path = commit / "receipt.json"
    scores_path = commit / "routed_scores.parquet"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("outcome_columns_consumed") not in (None, []):
        raise ValueError("staged score commit is not target-free")
    scores = pd.read_parquet(scores_path)
    feature = pd.read_parquet(args.e2_features.resolve())
    keys = ["candidate_id", "__decision_ts__", "__symbol__"]
    for frame in (scores, feature):
        frame["candidate_id"] = frame["candidate_id"].astype(str)
        frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
        frame["__symbol__"] = frame["__symbol__"].astype(str)
        if frame["candidate_id"].duplicated().any():
            raise ValueError("input carries duplicate candidate identities")
    # The live score commit normally has no 15-minute overlay columns.  A
    # historical parity commit can carry stale copies, however.  The dedicated
    # just-materialised target-free E2 panel is authoritative for every
    # feature-bearing overlap; never let an upstream cached copy shadow it.
    feature_columns = set(feature.columns).difference(keys)
    overlap = set(scores.columns).intersection(feature_columns)
    allowed_overlap = set(FIFTEEN_MINUTE_FEATURE_KEYS).union(VWAP_15M_FEATURE_KEYS).union(
        {"e2_feature_source_status", "e2_signal_atr", "e2_finite_feature_count"}
    )
    unexpected = sorted(overlap.difference(allowed_overlap))
    if unexpected:
        raise ValueError(f"score and E2 feature inputs overlap on non-E2 fields: {unexpected}")
    if overlap:
        scores = scores.drop(columns=sorted(overlap))
    joined = scores.merge(feature, on=keys, how="left", validate="one_to_one")
    if len(joined) != len(scores):
        raise AssertionError("E2 target-free feature join changed candidate population")
    selected, pairs = apply_e2_replacement(joined, bundle=bundle)
    output.mkdir(parents=True, exist_ok=False)
    selected.to_parquet(output / "e2_candidate_selection_target_free.parquet", index=False, compression="zstd")
    pairs.to_parquet(output / "e2_pair_predictions_target_free.parquet", index=False, compression="zstd")
    audit = {
        "schema": "strict_r3_p8u_e2_live_parity_noorder_score_v1",
        "status": "pass_target_free_no_order_e2_selection",
        "order_submission": False,
        "bundle": str(args.bundle.resolve()),
        "bundle_manifest_sha256": bundle.manifest_sha256,
        "staged_commit": str(commit),
        "staged_commit_receipt_sha256": _sha256(receipt_path),
        "e2_features": str(args.e2_features.resolve()),
        "e2_features_sha256": _sha256(args.e2_features.resolve()),
        "candidate_rows": int(len(selected)),
        "e2_source_complete_rows": int(selected.get("e2_feature_source_status", pd.Series(dtype=str)).isin(["ok", "complete"]).sum()),
        "ordinary_selected_rows": int(selected.e2_action.eq("ordinary_bcf_top2").sum()),
        "e2_replacements": int(selected.e2_action.eq("e2_q50_agreement_replacement").sum()),
        "e2_selected_rows": int(selected.e2_entry_selected.sum()),
        "outcome_columns_consumed": [],
        "next_required_stage": "normal constrained portfolio auction over e2_entry_selected rows only",
    }
    (output / "receipt.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
