#!/usr/bin/env python3
"""Build one causal resolved-label state for an hourly strict-R3 cycle.

The sealed exact-policy reserve is the immutable anchor.  Newly scored
activation-prefix candidates are joined to exact 15-minute policy labels only
after scoring, and only labels available before the current UTC day are
appended.  Current/unresolved paths can therefore never enter the EV map.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


LABEL_FIELDS = (
    "candidate_id", "policy_path_valid", "policy_gross_bps", "policy_net_bps",
    "policy_exit_bar_15m", "policy_exit_reason", "policy_entry_price",
    "policy_exit_price", "policy_label_available_ts", "policy_cost_bps",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _unique(frame: pd.DataFrame, name: str) -> None:
    if frame["candidate_id"].isna().any() or frame["candidate_id"].duplicated().any():
        raise ValueError(f"{name} has null or duplicate candidate identities")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-resolved-ledger", type=Path, required=True)
    parser.add_argument("--current-predictions", type=Path, required=True)
    parser.add_argument("--current-policy-labels", type=Path)
    parser.add_argument("--policy-json", type=Path, required=True)
    parser.add_argument("--decision-ts", required=True)
    parser.add_argument(
        "--intraday-frozen-ledger", type=Path,
        help=(
            "Exact prior hourly runtime ledger to carry forward unchanged. "
            "It must have the same UTC-day calibration cutoff."
        ),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable runtime resolved state exists: {args.out_dir}")

    base = pd.read_parquet(args.base_resolved_ledger)
    predictions = pd.read_parquet(args.current_predictions)
    for frame, name in ((base, "base ledger"), (predictions, "predictions")):
        frame["candidate_id"] = frame["candidate_id"].astype(str)
        _unique(frame, name)
    decision = _utc(args.decision_ts)
    day = decision.normalize()
    if args.intraday_frozen_ledger is not None:
        prior_path = args.intraday_frozen_ledger
        prior_manifest_path = prior_path.parent / "run_manifest.json"
        if not prior_path.exists() or not prior_manifest_path.exists():
            raise FileNotFoundError(
                "intraday calibration carry requires the prior ledger and manifest"
            )
        prior_manifest = json.loads(prior_manifest_path.read_text())
        prior_decision = _utc(prior_manifest["decision_ts"])
        prior_cutoff = _utc(prior_manifest["calibration_cutoff_utc_day"])
        if prior_decision >= decision:
            raise ValueError("intraday calibration source must precede the decision")
        if prior_cutoff != day or prior_decision.normalize() != day:
            raise ValueError(
                "intraday calibration carry may not cross a UTC-day boundary"
            )
        prior = pd.read_parquet(prior_path)
        prior["policy_label_available_ts"] = pd.to_datetime(
            prior["policy_label_available_ts"], utc=True, errors="raise",
        )
        if not prior["policy_label_available_ts"].lt(day).all():
            raise AssertionError(
                "intraday calibration source contains labels unavailable at day start"
            )
        if str(prior_manifest.get("policy_json_sha256")) != _sha(args.policy_json):
            raise ValueError("intraday calibration source uses another exit policy")
        args.out_dir.mkdir(parents=True)
        output_path = args.out_dir / "walkforward_scored_label_ledger.parquet"
        shutil.copyfile(prior_path, output_path)
        if _sha(output_path) != _sha(prior_path):
            raise AssertionError("intraday calibration ledger copy is not byte exact")
        manifest = {
            "schema": "strict_r3_runtime_resolved_calibration_state_v1",
            "decision_ts": decision.isoformat(),
            "calibration_cutoff_utc_day": day.isoformat(),
            "base_rows": int(len(prior)),
            "newly_appended_rows": 0,
            "previously_invalid_rows_repaired": 0,
            "overlap_rows_skipped_by_base_route": 0,
            "rows": int(len(prior)),
            "max_label_available_ts": (
                prior["policy_label_available_ts"].max().isoformat()
                if len(prior) else None
            ),
            "strictly_prior_to_utc_day": True,
            "current_decision_ids_appended": 0,
            "intraday_state_frozen": True,
            "intraday_source_decision_ts": prior_decision.isoformat(),
            "intraday_source_ledger_sha256": _sha(prior_path),
            "policy_json": str(args.policy_json),
            "policy_json_sha256": _sha(args.policy_json),
            "sources": {"intraday_frozen_ledger": _sha(prior_path)},
            "output": str(output_path),
            "output_sha256": _sha(output_path),
        }
        (args.out_dir / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        print(json.dumps({"event": "complete", **manifest}))
        return
    if args.current_policy_labels is None:
        raise ValueError(
            "--current-policy-labels is required outside intraday frozen carry"
        )
    labels = pd.read_parquet(args.current_policy_labels, columns=list(LABEL_FIELDS))
    labels["candidate_id"] = labels["candidate_id"].astype(str)
    _unique(labels, "labels")
    base["policy_label_available_ts"] = pd.to_datetime(
        base["policy_label_available_ts"], utc=True, errors="raise",
    )
    # The sealed ledger may contain labels that resolved later in its archived
    # evaluation period. They are provenance, not information available now.
    # Remove them from the runtime state before any overlap or mapping logic.
    base = base.loc[base["policy_label_available_ts"].lt(day)].copy()
    predictions["__decision_ts__"] = pd.to_datetime(
        predictions["__decision_ts__"], utc=True, errors="raise",
    )
    labels["policy_label_available_ts"] = pd.to_datetime(
        labels["policy_label_available_ts"], utc=True, errors="raise",
    )
    current = predictions.merge(labels, on="candidate_id", how="inner", validate="one_to_one")
    resolved_now = (
        current["policy_label_available_ts"].lt(day)
        & current["policy_path_valid"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(current["policy_net_bps"], errors="coerce"))
    )
    current = current.loc[resolved_now].copy()
    if len(current) and not np.allclose(
        current["policy_net_bps"].to_numpy(float),
        current["policy_gross_bps"].to_numpy(float) - 100.0,
        atol=1e-9, rtol=0.0,
    ):
        raise ValueError("runtime policy labels do not subtract 100-bps cost exactly once")

    overlap = set(base["candidate_id"]).intersection(current["candidate_id"])
    repaired_invalid_rows = 0
    overlap_rows_skipped_by_route = 0
    if overlap:
        left = base.loc[base["candidate_id"].isin(overlap)].set_index("candidate_id").sort_index()
        right = current.loc[current["candidate_id"].isin(overlap)].set_index("candidate_id").sort_index()
        for field in (
            "final_score", "conversion_bundle_sha256", "upstream_bundle_sha256",
            "geometry_bundle_sha256", "policy_label_available_ts",
        ):
            if field not in left or field not in right:
                raise ValueError(f"runtime overlap audit lacks {field}")
            if field == "final_score":
                # Schema-v6 stops after the base for candidates outside the
                # timestamp-local top-20% route.  A prefix replay therefore
                # returns null downstream scores for those rows even when the
                # sealed historical ledger predates the compute gate and
                # contains their formerly materialised scores.  Compare every
                # score that the current bundle actually recomputed, retain the
                # immutable resolved row for the skipped population, and never
                # treat a route-induced null as a new score vintage.
                recomputed = pd.to_numeric(right[field], errors="coerce").notna()
                if "base_route_eligible" in right:
                    routed = right["base_route_eligible"].fillna(False).astype(bool)
                    if not recomputed.eq(routed).all():
                        raise ValueError(
                            "runtime overlap final-score availability disagrees "
                            "with base_route_eligible"
                        )
                overlap_rows_skipped_by_route = int((~recomputed).sum())
                if not np.allclose(
                    pd.to_numeric(left.loc[recomputed, field]).to_numpy(float),
                    pd.to_numeric(right.loc[recomputed, field]).to_numpy(float),
                    atol=1e-9, rtol=0.0,
                    equal_nan=True,
                ):
                    raise ValueError(f"runtime resolved state conflicts on {field}")
            elif not left[field].astype(str).eq(right[field].astype(str)).all():
                raise ValueError(f"runtime resolved state conflicts on {field}")
        left_valid = left["policy_path_valid"].fillna(False).astype(bool)
        right_valid = right["policy_path_valid"].fillna(False).astype(bool)
        both_valid = left_valid & right_valid
        if both_valid.any() and not np.allclose(
            pd.to_numeric(left.loc[both_valid, "policy_net_bps"]).to_numpy(float),
            pd.to_numeric(right.loc[both_valid, "policy_net_bps"]).to_numpy(float),
            atol=1e-9, rtol=0.0,
        ):
            raise ValueError("runtime resolved state conflicts on policy_net_bps")
        repair_ids = left.index[~left_valid & right_valid]
        repaired_invalid_rows = int(len(repair_ids))
        if repaired_invalid_rows:
            replacement = right.loc[repair_ids, list(LABEL_FIELDS[1:])].copy()
            indexed = base.set_index("candidate_id")
            for field in LABEL_FIELDS[1:]:
                indexed.loc[repair_ids, field] = replacement.loc[repair_ids, field]
            base = indexed.reset_index()
        current = current.loc[~current["candidate_id"].isin(overlap)].copy()

    # Only completed routed-stack rows are eligible calibration observations.
    # Base-only rows remain visible in the scoring/rejection audit but must not
    # enter the resolved EV/correctness state with a missing final score.
    current = current.loc[
        pd.to_numeric(current.get("final_score"), errors="coerce").notna()
    ].copy()
    output = pd.concat([base, current.reindex(columns=base.columns)], ignore_index=True)
    _unique(output, "runtime resolved output")
    output["policy_label_available_ts"] = pd.to_datetime(
        output["policy_label_available_ts"], utc=True, errors="raise",
    )
    if not output.loc[
        output["__decision_ts__"].ge(pd.to_datetime(predictions["__decision_ts__"].min(), utc=True)),
        "policy_label_available_ts",
    ].lt(day).all():
        raise AssertionError("runtime state appended a label unavailable at UTC day start")
    args.out_dir.mkdir(parents=True)
    output_path = args.out_dir / "walkforward_scored_label_ledger.parquet"
    output.to_parquet(output_path, index=False, compression="zstd")
    policy_manifest = args.current_policy_labels.parent / "run_manifest.json"
    manifest = {
        "schema": "strict_r3_runtime_resolved_calibration_state_v1",
        "decision_ts": decision.isoformat(),
        "calibration_cutoff_utc_day": day.isoformat(),
        "base_rows": int(len(base)),
        "newly_appended_rows": int(len(current)),
        "previously_invalid_rows_repaired": repaired_invalid_rows,
        "overlap_rows_skipped_by_base_route": overlap_rows_skipped_by_route,
        "rows": int(len(output)),
        "max_label_available_ts": (
            output["policy_label_available_ts"].max().isoformat() if len(output) else None
        ),
        "strictly_prior_to_utc_day": bool(
            output.loc[
                output["__decision_ts__"].ge(pd.to_datetime(predictions["__decision_ts__"].min(), utc=True)),
                "policy_label_available_ts",
            ].lt(day).all()
        ),
        "current_decision_ids_appended": int(
            current["__decision_ts__"].eq(decision).sum()
        ),
        "policy_json": str(args.policy_json),
        "policy_json_sha256": _sha(args.policy_json),
        "sources": {
            "base_resolved_ledger": _sha(args.base_resolved_ledger),
            "current_predictions": _sha(args.current_predictions),
            "current_policy_labels": _sha(args.current_policy_labels),
            "current_policy_manifest": _sha(policy_manifest),
        },
        "output": str(output_path),
        "output_sha256": _sha(output_path),
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", **manifest}))


if __name__ == "__main__":
    main()
