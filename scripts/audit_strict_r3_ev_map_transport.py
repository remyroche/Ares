#!/usr/bin/env python3
"""Audit whether strict-R3 score-to-policy-net mapping transports by producer.

The admission score is a causal CDF, but its expected-value curve must still
be empirically stable when either the conversion bundle or the monthly
upstream base/meta bundle changes.  This read-only diagnostic compares every
producer's *strict-OOF realised* score-bin economics with the previously
resolved producer population.  It never modifies scores, labels or admission
decisions; its output is an explicit gate for any future cross-producer
42-day EV-map bridge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _top(frame: pd.DataFrame, fraction: float) -> pd.DataFrame:
    ordered = frame.sort_values(
        ["final_score", "candidate_id"], ascending=[False, True], kind="stable",
    )
    return ordered.head(max(1, int(np.ceil(float(fraction) * len(ordered))))).copy()


def _mean(frame: pd.DataFrame) -> float:
    return float(pd.to_numeric(frame["policy_net_bps"], errors="coerce").mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored-ledger", type=Path, required=True)
    parser.add_argument("--producer-lineage", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-top5-support", type=int, default=500)
    parser.add_argument("--max-top5-drift-bps", type=float, default=50.0)
    parser.add_argument("--max-score-ks", type=float, default=0.25)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable transport audit already exists: {args.out_dir}")
    if args.min_top5_support < 4:
        raise ValueError("--min-top5-support must be at least four")

    ledger_columns = [
        "candidate_id", "__decision_ts__", "final_score", "policy_net_bps",
        "policy_label_available_ts", "policy_path_valid", "stack_is_prequential",
        "conversion_bundle_sha256", "geometry_bundle_sha256",
    ]
    lineage_columns = [
        "candidate_id", "upstream_bundle_sha256", "ev_score_family_id",
    ]
    ledger = pd.read_parquet(args.scored_ledger, columns=ledger_columns)
    lineage = pd.read_parquet(args.producer_lineage, columns=lineage_columns)
    if ledger["candidate_id"].duplicated().any() or lineage["candidate_id"].duplicated().any():
        raise ValueError("transport audit requires unique candidate IDs")
    frame = ledger.merge(lineage, on="candidate_id", how="inner", validate="one_to_one")
    if len(frame) != len(ledger):
        raise ValueError("producer lineage does not cover the entire score ledger")
    if not frame["stack_is_prequential"].fillna(False).astype(bool).all():
        raise ValueError("transport audit requires strict prequential score rows")
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True)
    frame["policy_label_available_ts"] = pd.to_datetime(
        frame["policy_label_available_ts"], utc=True,
    )
    frame["final_score"] = pd.to_numeric(frame["final_score"], errors="coerce")
    frame["policy_net_bps"] = pd.to_numeric(frame["policy_net_bps"], errors="coerce")
    frame = frame.loc[
        frame["policy_path_valid"].fillna(False).astype(bool)
        & np.isfinite(frame["final_score"])
        & np.isfinite(frame["policy_net_bps"])
    ].copy()
    if frame.empty:
        raise ValueError("no valid strict-OOF policy outcomes remain")
    producer_columns = [
        "ev_score_family_id", "conversion_bundle_sha256",
        "upstream_bundle_sha256", "geometry_bundle_sha256",
    ]
    frame["producer_id"] = frame[producer_columns].astype(str).agg("|".join, axis=1)
    groups = sorted(
        (
            (producer, block.sort_values(["__decision_ts__", "candidate_id"], kind="stable"))
            for producer, block in frame.groupby("producer_id", sort=False)
        ),
        key=lambda item: item[1]["__decision_ts__"].min(),
    )

    producer_rows: list[dict[str, object]] = []
    transport_rows: list[dict[str, object]] = []
    prior = frame.iloc[:0].copy()
    for index, (producer, block) in enumerate(groups):
        start = block["__decision_ts__"].min()
        end = block["__decision_ts__"].max() + pd.Timedelta(hours=1)
        top1 = _top(block, 0.01)
        top5 = _top(block, 0.05)
        producer_rows.append({
            "producer_id": producer,
            "producer_index": index,
            "start": start,
            "end_exclusive": end,
            "rows": int(len(block)),
            "top1_rows": int(len(top1)),
            "top5_rows": int(len(top5)),
            "score_p50": float(block["final_score"].median()),
            "score_p95": float(block["final_score"].quantile(0.95)),
            "net_all_bps": _mean(block),
            "net_top1_bps": _mean(top1),
            "net_top5_bps": _mean(top5),
            "score_net_spearman": float(
                spearmanr(block["final_score"], block["policy_net_bps"]).statistic,
            ),
            **{column: str(block[column].iloc[0]) for column in producer_columns},
        })
        # At this producer boundary, a live bridge could only use earlier
        # labels resolved before the first candidate decision.
        reference = prior.loc[prior["policy_label_available_ts"].lt(start)].copy()
        if reference.empty:
            transport_rows.append({
                "producer_id": producer,
                "producer_index": index,
                "reference_rows": 0,
                "bridge_status": "insufficient_prior_resolved_support",
                "bridge_pass": False,
            })
        else:
            reference_top5 = _top(reference, 0.05)
            ks = float(ks_2samp(
                block["final_score"].to_numpy(float),
                reference["final_score"].to_numpy(float),
                method="asymp",
            ).statistic)
            child_top5 = _mean(top5)
            parent_top5 = _mean(reference_top5)
            drift = child_top5 - parent_top5
            support_ok = len(top5) >= int(args.min_top5_support)
            sign_ok = (child_top5 >= 0.0) == (parent_top5 >= 0.0)
            drift_ok = abs(drift) <= float(args.max_top5_drift_bps)
            distribution_ok = ks <= float(args.max_score_ks)
            passed = bool(support_ok and sign_ok and drift_ok and distribution_ok)
            transport_rows.append({
                "producer_id": producer,
                "producer_index": index,
                "reference_rows": int(len(reference)),
                "reference_top5_rows": int(len(reference_top5)),
                "child_top5_rows": int(len(top5)),
                "reference_top5_net_bps": parent_top5,
                "child_top5_net_bps": child_top5,
                "top5_drift_bps": drift,
                "score_distribution_ks": ks,
                "support_ok": support_ok,
                "sign_ok": sign_ok,
                "top5_drift_ok": drift_ok,
                "score_distribution_ok": distribution_ok,
                "bridge_status": "pass" if passed else "reject_cross_producer_pooling",
                "bridge_pass": passed,
            })
        prior = pd.concat([prior, block], ignore_index=True)

    producer_metrics = pd.DataFrame(producer_rows)
    transport = pd.DataFrame(transport_rows)
    evaluable = transport.loc[transport["reference_rows"].gt(0)].copy()
    decision = {
        "schema": "strict_r3_ev_map_transport_audit_v1",
        "producer_count": int(len(producer_metrics)),
        "evaluable_transition_count": int(len(evaluable)),
        "passing_transition_count": int(evaluable["bridge_pass"].sum()),
        "cross_producer_42d_pooling_approved": bool(
            len(evaluable) > 0 and evaluable["bridge_pass"].all(),
        ),
        "gate": {
            "min_top5_support": int(args.min_top5_support),
            "max_top5_drift_bps": float(args.max_top5_drift_bps),
            "max_score_ks": float(args.max_score_ks),
            "requires_top5_sign_agreement": True,
            "reference": "only earlier producer rows with label_available_ts < child producer start",
        },
        "conclusion": (
            "CROSS_PRODUCER_42D_POOLING_APPROVED"
            if len(evaluable) > 0 and evaluable["bridge_pass"].all()
            else "CROSS_PRODUCER_42D_POOLING_REJECTED_USE_FULL_PRODUCER_FAIL_CLOSED"
        ),
    }
    args.out_dir.mkdir(parents=True)
    producer_metrics.to_parquet(
        args.out_dir / "producer_score_economics.parquet", index=False,
        compression="zstd",
    )
    transport.to_parquet(
        args.out_dir / "producer_transport_gate.parquet", index=False,
        compression="zstd",
    )
    (args.out_dir / "transport_decision.json").write_text(
        json.dumps(decision, indent=2) + "\n",
    )
    manifest = {
        **decision,
        "scored_ledger": str(args.scored_ledger),
        "scored_ledger_sha256": _sha(args.scored_ledger),
        "producer_lineage": str(args.producer_lineage),
        "producer_lineage_sha256": _sha(args.producer_lineage),
        "scores_recomputed": False,
        "outcome_columns_consumed_for_diagnostic_only": ["policy_net_bps"],
    }
    (args.out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
    )
    print(json.dumps({"event": "complete", "output": str(args.out_dir), **decision}))


if __name__ == "__main__":
    main()
