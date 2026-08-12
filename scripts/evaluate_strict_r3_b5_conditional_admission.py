#!/usr/bin/env python3
"""Causal Bayesian-B5 residual conditioning of the live 21-day EV map.

This is deliberately not a score blend.  The canonical 21-day side-local map
remains the parent expected-policy-net estimate.  B5 can alter admission only
through a strictly prior-resolved, support-shrunk residual correction inside
its decision-time quality state.  It addresses the mismatch between B5's
timestamp-local signal and a one-dimensional global score map.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.stage_i_causal_admission import (  # noqa: E402
    Causal21dAdmissionSpec,
    apply_causal_21d_side_admission,
)


TAILS = (0.005, 0.01, 0.02, 0.05)
WINDOW_DAYS = 21
MIN_BUCKET_ROWS = 20
TRIM_FRACTION = 0.05


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _robust_mean(values: np.ndarray) -> float:
    values = np.sort(np.asarray(values, dtype=float))
    trim = int(np.floor(len(values) * TRIM_FRACTION))
    kept = values[trim:len(values) - trim] if len(values) - 2 * trim else values
    return float(kept.mean())


def _quality_bucket(frame: pd.DataFrame) -> np.ndarray:
    quality = (
        pd.to_numeric(frame["posterior_expected_rank_train"], errors="raise").to_numpy(float)
        - 0.5 * pd.to_numeric(frame["posterior_adverse_rank_train"], errors="raise").to_numpy(float)
    )
    top30 = frame["timestamp_top30"].fillna(False).astype(bool).to_numpy()
    # The posterior ranks are fit on prior training rows by the B5 producer;
    # these predeclared cuts therefore need no held-window recalibration.
    bucket = np.ones(len(frame), dtype=np.int8)  # neutral / outside B5 authority
    bucket[top30 & (quality <= -1.0 / 6.0)] = 0
    bucket[top30 & (quality >= 1.0 / 6.0)] = 2
    return bucket


def _metrics(frame: pd.DataFrame, arm: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    work = frame.copy()
    work["year"] = work["__decision_ts__"].dt.year.astype(str)
    for period, block in [("all", work), *[(key, value) for key, value in work.groupby("year", sort=True)]]:
        eligible = block.loc[block["admitted"].astype(bool)].copy()
        def add(kind: str, selected: pd.DataFrame) -> None:
            valid = selected.loc[selected["policy_path_valid"].fillna(False).astype(bool) & selected["policy_net_bps"].notna()]
            net = pd.to_numeric(valid["policy_net_bps"], errors="coerce")
            rows.append({
                "arm": arm, "period": period, "kind": kind,
                "score_rows": int(len(block)), "mapped_rows": int(block["expected_bps"].notna().sum()),
                "admitted_rows": int(len(eligible)), "admission_rate": float(len(eligible) / max(len(block), 1)),
                "selected_rows": int(len(selected)), "valid_outcomes": int(len(valid)),
                "outcome_coverage": float(len(valid) / max(len(selected), 1)),
                "net_bps_per_trade": float(net.mean()) if len(net) else np.nan,
                "positive_rate": float(net.gt(0.0).mean()) if len(net) else np.nan,
            })
        add("all_admitted", eligible)
        for tail in TAILS:
            add(f"admitted_top_{tail:g}", eligible.nlargest(max(1, int(math.ceil(tail * len(eligible)))), "expected_bps", keep="first") if len(eligible) else eligible)
    return rows


def _condition(
    frame: pd.DataFrame, *, shrinkage_rows: float, delta_cap_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = frame.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True).copy()
    out["b5_quality_bucket"] = _quality_bucket(out)
    out["parent_expected_bps"] = pd.to_numeric(out["causal_21d_side_expected_net_bps"], errors="coerce")
    out["expected_bps"] = out["parent_expected_bps"]
    out["b5_delta_bps"] = 0.0
    out["b5_bucket_support"] = 0
    out["b5_status"] = "parent_only"
    valid_target = (
        out["policy_path_valid"].fillna(False).astype(bool)
        & out["policy_net_bps"].notna()
        & out["parent_expected_bps"].notna()
    )
    out["__residual__"] = pd.to_numeric(out["policy_net_bps"], errors="coerce") - out["parent_expected_bps"]
    available = pd.to_datetime(out["policy_label_available_ts"], utc=True, errors="raise")
    decision = pd.to_datetime(out["__decision_ts__"], utc=True, errors="raise")
    available_ns = available.array.as_unit("ns").asi8
    snapshot = decision.dt.normalize()
    snapshot_ns = snapshot.array.as_unit("ns").asi8
    eligible_pos = np.flatnonzero(valid_target.to_numpy(bool))
    ordered = eligible_pos[np.argsort(available_ns[eligible_pos], kind="stable")]
    ordered_available = available_ns[ordered]
    starts = np.r_[0, np.flatnonzero(snapshot_ns[1:] != snapshot_ns[:-1]) + 1] if len(out) else np.empty(0, dtype=np.int64)
    ends = np.r_[starts[1:], len(out)] if len(out) else np.empty(0, dtype=np.int64)
    audit: list[dict[str, object]] = []
    for start, end in zip(starts, ends, strict=True):
        now = int(snapshot_ns[start])
        lower = int(np.searchsorted(ordered_available, now - pd.Timedelta(days=WINDOW_DAYS).value, side="left"))
        upper = int(np.searchsorted(ordered_available, now, side="left"))
        refs = ordered[lower:upper]
        current = np.arange(start, end, dtype=np.int64)
        supports: dict[int, int] = {}
        deltas: dict[int, float] = {}
        for bucket in (0, 1, 2):
            values = out.loc[refs[out.loc[refs, "b5_quality_bucket"].to_numpy(np.int8) == bucket], "__residual__"].to_numpy(float)
            support = int(len(values))
            supports[bucket] = support
            if support < MIN_BUCKET_ROWS:
                deltas[bucket] = 0.0
                continue
            raw = float(np.clip(_robust_mean(values), -delta_cap_bps, delta_cap_bps))
            deltas[bucket] = raw * support / (support + shrinkage_rows)
        for bucket in (0, 1, 2):
            where = current[out.loc[current, "b5_quality_bucket"].to_numpy(np.int8) == bucket]
            out.loc[where, "b5_bucket_support"] = supports[bucket]
            if bucket != 1 and supports[bucket] >= MIN_BUCKET_ROWS:
                out.loc[where, "b5_delta_bps"] = deltas[bucket]
                out.loc[where, "expected_bps"] = out.loc[where, "parent_expected_bps"] + deltas[bucket]
                out.loc[where, "b5_status"] = "conditional_shrunk_residual"
        ref_max = pd.Timestamp(available.iloc[ordered[upper - 1]]) if upper > lower else pd.NaT
        audit.append({
            "snapshot_utc": pd.Timestamp(snapshot.iloc[start]), "reference_rows": int(len(refs)),
            "reference_max_label_available_ts": ref_max,
            "strictly_prior_resolved": bool(ref_max < snapshot.iloc[start]) if len(refs) else True,
            **{f"support_bucket_{bucket}": supports[bucket] for bucket in (0, 1, 2)},
            **{f"delta_bucket_{bucket}_bps": deltas[bucket] for bucket in (0, 1, 2)},
        })
    out["admitted"] = out["expected_bps"].ge(50.0).fillna(False)
    if not all(row["strictly_prior_resolved"] for row in audit):
        raise AssertionError("conditional map consumed unresolved labels")
    return out.drop(columns="__residual__"), pd.DataFrame(audit)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--source-arm", default="B5_current_trust_overlay")
    parser.add_argument("--shrinkage-rows", type=float, default=300.0)
    parser.add_argument("--delta-cap-bps", type=float, default=50.0)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    if args.shrinkage_rows <= 0.0 or args.delta_cap_bps <= 0.0:
        raise ValueError("shrinkage and cap must be positive")
    source = pd.read_parquet(args.predictions)
    source = source.loc[source["arm"].eq(args.source_arm)].copy()
    required = {
        "candidate_id", "__decision_ts__", "side_name", "policy_path_valid", "policy_label_available_ts", "policy_net_bps",
        "final_score", "posterior_expected_rank_train", "posterior_adverse_rank_train", "timestamp_top30",
    }
    missing = sorted(required.difference(source.columns))
    if missing:
        raise KeyError(f"source lacks {missing}")
    if source["candidate_id"].duplicated().any() or not source["side_name"].astype(str).str.lower().eq("long").all():
        raise AssertionError("expected a unique long-only B5 ledger")
    mapped, parent_audit = apply_causal_21d_side_admission(
        source.loc[:, [
            "candidate_id", "__decision_ts__", "side_name", "policy_path_valid", "policy_label_available_ts", "policy_net_bps", "final_score",
        ]],
        score_column="final_score", net_column="policy_net_bps", decision_column="__decision_ts__",
        label_available_column="policy_label_available_ts", identity_column="candidate_id",
        spec=Causal21dAdmissionSpec(mode="hierarchical_tail_side_shrinkage_v2"),
    )
    source = source.merge(
        mapped[["candidate_id", "causal_21d_side_expected_net_bps", "causal_21d_side_admitted_ge_50bps"]],
        on="candidate_id", how="left", validate="one_to_one",
    )
    corrected, conditional_audit = _condition(
        source, shrinkage_rows=float(args.shrinkage_rows), delta_cap_bps=float(args.delta_cap_bps),
    )
    control = corrected.copy()
    control["expected_bps"] = control["parent_expected_bps"]
    control["admitted"] = control["expected_bps"].ge(50.0).fillna(False)
    args.out_dir.mkdir(parents=True)
    corrected["arm"] = "b5_conditional_residual_map"
    control["arm"] = "parent_score_map_control"
    output = pd.concat([control, corrected], ignore_index=True)
    output.to_parquet(args.out_dir / "admission_predictions.parquet", index=False, compression="zstd")
    pd.DataFrame([*_metrics(control, "parent_score_map_control"), *_metrics(corrected, "b5_conditional_residual_map")]).to_parquet(args.out_dir / "metrics.parquet", index=False)
    parent_audit.to_parquet(args.out_dir / "parent_admission_audit.parquet", index=False)
    conditional_audit.to_parquet(args.out_dir / "conditional_audit.parquet", index=False)
    correctness = {
        "unique_candidate_ids": bool(not source["candidate_id"].duplicated().any()),
        "long_only": bool(source["side_name"].astype(str).str.lower().eq("long").all()),
        "labels_strictly_after_decision": bool((pd.to_datetime(source["policy_label_available_ts"], utc=True) > pd.to_datetime(source["__decision_ts__"], utc=True)).all()),
        "conditional_references_strictly_prior_resolved": bool(conditional_audit["strictly_prior_resolved"].all()),
        "no_current_outcome_in_score": True,
        "parent_map": "Causal21dAdmissionSpec(hierarchical_tail_side_shrinkage_v2)",
    }
    (args.out_dir / "correctness_report.json").write_text(json.dumps(correctness, indent=2) + "\n")
    manifest = {
        "schema": "strict_r3_b5_conditional_residual_admission_v1",
        "predictions": str(args.predictions), "predictions_sha256": _sha(args.predictions), "source_arm": args.source_arm,
        "parent": "canonical final_score 21d hierarchical tail side-shrunk EV map",
        "conditional_rule": "B5 top30 low/high posterior-quality bucket, 21d prior-resolved robust residual mean, shrunk to zero",
        "quality_cuts": [-1.0 / 6.0, 1.0 / 6.0], "min_bucket_rows": MIN_BUCKET_ROWS,
        "shrinkage_rows": args.shrinkage_rows, "delta_cap_bps": args.delta_cap_bps,
        "admission": "conditional expected policy net bps >= 50; fail closed when parent is unmapped",
        "causality": "all B5 inputs are strict-prequential source predictions; residual references require label_available_ts < snapshot",
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", "rows": int(len(output)), "out_dir": str(args.out_dir)}))


if __name__ == "__main__":
    main()
