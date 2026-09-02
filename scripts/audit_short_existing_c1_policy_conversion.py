#!/usr/bin/env python3
"""Audit frozen C1 predictions against the complete exact-policy label ledger."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_short_policy_conversion_funnel import (
    _global_tails,
    _load_policy_ledger,
    _within_query_scorecard,
    _utc,
)


def run(*, out: Path, predictions_path: Path, policies: Path, start: pd.Timestamp, end: pd.Timestamp) -> Path:
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    predictions = pd.read_parquet(predictions_path, columns=["candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name", "score"])
    for column in ("__ts__", "__decision_ts__"):
        predictions[column] = pd.to_datetime(predictions[column], utc=True, errors="raise")
    predictions = predictions.loc[predictions.__ts__.ge(start) & predictions.__ts__.lt(end)].copy()
    labels = _load_policy_ledger(policies, start, end)
    labels = labels.loc[:, ["candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name", "policy_path_valid", "p0_canonical_net_bps", "p0_canonical_gross_bps"]]
    frame = predictions.merge(labels, on=["candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name"], how="left", validate="one_to_one")
    if len(frame) != len(predictions) or frame.policy_path_valid.isna().any():
        raise AssertionError("existing C1 prediction and policy-ledger identities do not match")
    frame["timestamp_percentile"] = frame.groupby("__ts__", sort=False).score.rank(method="average", pct=True)
    values: list[dict] = []
    deciles: list[dict] = []
    for column in ("score", "timestamp_percentile"):
        metric, bins = _within_query_scorecard(frame, score_column=column, spec="C1_existing_ordinary", scope="oos")
        values.extend(metric); deciles.extend(bins)
    pd.DataFrame(values).to_parquet(out / "within_timestamp_scorecard.parquet", index=False, compression="zstd")
    pd.DataFrame(deciles).to_parquet(out / "within_timestamp_deciles.parquet", index=False, compression="zstd")
    pd.DataFrame(_global_tails(frame, spec="C1_existing_ordinary", scope="oos")).to_parquet(out / "global_policy_tail_metrics.parquet", index=False, compression="zstd")
    (out / "run_manifest.json").write_text(json.dumps({
        "schema": "strict_r3_short_existing_c1_policy_conversion_audit_v1", "status": "complete",
        "prediction_path": str(predictions_path), "policy_ledger": str(policies),
        "window": f"[{start.isoformat()}, {end.isoformat()})", "scored_rows": int(len(frame)),
        "policy_valid_rows": int(frame.policy_path_valid.sum()),
        "raw_vs_timestamp_percentile": "both are reported; per-query ordering is expected to agree except score ties",
    }, indent=2) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--policies", type=Path, required=True)
    parser.add_argument("--start", default="2024-10-01T00:00:00Z")
    parser.add_argument("--end", default="2025-01-01T00:00:00Z")
    args = parser.parse_args()
    print(run(out=args.out.resolve(), predictions_path=args.predictions.resolve(), policies=args.policies.resolve(), start=_utc(args.start), end=_utc(args.end)))


if __name__ == "__main__":
    main()
