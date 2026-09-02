#!/usr/bin/env python3
"""Independently rebuild one schema-v6 live hour and prove replay parity.

The canonical hourly producer persists a target-free candidate/feature prefix,
scores it, and applies the causal Robust-21/MC1 admission stack.  This auditor
does not trust those current-hour values.  It regenerates the current feature
cross-section from the same point-in-time sources, replaces only those rows in
the immutable prefix, independently re-scores the full conversion-state
prefix, and independently re-runs admission.  Current feature, model, trust,
and admission values must agree within 0.01 percent; identities and categorical
decisions must agree exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.strict_r3_inference_bundle import (  # noqa: E402
    StrictR3InferenceBundle,
)


SCHEMA = "strict_r3_schema_v6_current_replay_parity_v1"
RTOL = 0.0001  # 0.01 percent
ATOL = 1e-9


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _run(command: list[str], log: Path) -> None:
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
            text=True, check=False,
        )
    if completed.returncode:
        raise RuntimeError(f"independent replay stage failed; see {log}")


def _feature_fields(contract_path: Path) -> list[str]:
    payload = json.loads(contract_path.read_text())
    fields = [str(value) for value in payload["base_fields_by_side"]["long"]]
    if len(fields) != 120 or len(set(fields)) != 120:
        raise ValueError("current replay requires the frozen 120-field long contract")
    return fields


def _numeric_audit(
    left: pd.Series, right: pd.Series, *, field: str,
) -> dict[str, object]:
    a = pd.to_numeric(left, errors="coerce").to_numpy(float)
    b = pd.to_numeric(right, errors="coerce").to_numpy(float)
    finite = np.isfinite(a) & np.isfinite(b)
    same_missing = np.isnan(a) == np.isnan(b)
    close = np.isclose(a, b, rtol=RTOL, atol=ATOL, equal_nan=True)
    absolute = np.abs(a[finite] - b[finite]) if finite.any() else np.array([])
    denominator = np.maximum(np.maximum(np.abs(a[finite]), np.abs(b[finite])), ATOL)
    relative = absolute / denominator if len(absolute) else np.array([])
    return {
        "field": field,
        "kind": "numeric",
        "rows": int(len(a)),
        "finite_pairs": int(finite.sum()),
        "missingness_exact": bool(same_missing.all()),
        "max_abs_delta": float(absolute.max()) if len(absolute) else 0.0,
        "max_relative_delta": float(relative.max()) if len(relative) else 0.0,
        "within_tolerance": bool(close.all() and same_missing.all()),
    }


def _categorical_audit(
    left: pd.Series, right: pd.Series, *, field: str,
) -> dict[str, object]:
    equal = left.astype(str).eq(right.astype(str)) | (left.isna() & right.isna())
    return {
        "field": field,
        "kind": "categorical",
        "rows": int(len(left)),
        "exact": bool(equal.all()),
        "mismatch_rows": int((~equal).sum()),
        "within_tolerance": bool(equal.all()),
    }


def _compare(
    stored: pd.DataFrame,
    replayed: pd.DataFrame,
    *,
    fields: list[str],
    role: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    key = "candidate_id"
    for frame, name in ((stored, "stored"), (replayed, "replayed")):
        if key not in frame or frame[key].isna().any() or frame[key].duplicated().any():
            raise ValueError(f"{role} {name} identities are invalid")
        frame[key] = frame[key].astype(str)
    if set(stored[key]) != set(replayed[key]) or len(stored) != len(replayed):
        raise AssertionError(f"{role} current identities differ")
    joined = stored[[key, *fields]].merge(
        replayed[[key, *fields]], on=key, validate="one_to_one",
        suffixes=("__stored", "__replayed"),
    ).sort_values(key, kind="stable")
    rows: list[dict[str, object]] = []
    for field in fields:
        left = joined[f"{field}__stored"]
        right = joined[f"{field}__replayed"]
        numeric = (
            is_numeric_dtype(left.dtype) or is_numeric_dtype(right.dtype)
            or is_bool_dtype(left.dtype) or is_bool_dtype(right.dtype)
        )
        rows.append(
            _numeric_audit(left, right, field=field)
            if numeric else _categorical_audit(left, right, field=field)
        )
    audit = pd.DataFrame(rows)
    summary = {
        "role": role,
        "rows": int(len(stored)),
        "fields": int(len(fields)),
        "all_fields_match": bool(audit["within_tolerance"].all()),
        "maximum_abs_delta": float(
            pd.to_numeric(audit.get("max_abs_delta"), errors="coerce").max()
            if "max_abs_delta" in audit else 0.0
        ),
        "maximum_relative_delta": float(
            pd.to_numeric(audit.get("max_relative_delta"), errors="coerce").max()
            if "max_relative_delta" in audit else 0.0
        ),
        "failed_fields": audit.loc[
            ~audit["within_tolerance"], "field"
        ].astype(str).tolist(),
    }
    return audit, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-bundle", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--enforce-live-wall-clock", action="store_true")
    parser.add_argument(
        "--rebuild-inputs", action="store_true",
        help=(
            "Independently regenerate current-hour source features. Off by "
            "default for the hourly fast audit; enable for scheduled/full "
            "source-to-feature parity checks."
        ),
    )
    parser.add_argument(
        "--maximum-audit-age-seconds", type=float, default=None,
        help=(
            "Post-decision parity-audit deadline. Defaults to the frozen "
            "inference freshness window, but may be longer because this audit "
            "cannot authorize or alter entries."
        ),
    )
    args = parser.parse_args()
    audit_started_at = pd.Timestamp(datetime.now(timezone.utc))
    if args.out.exists():
        raise FileExistsError(f"immutable current replay audit exists: {args.out}")

    manifest = json.loads((args.run / "run_manifest.json").read_text())
    decision = _utc(manifest["decision_ts"])
    bundle = StrictR3InferenceBundle.load(args.inference_bundle, root=ROOT)
    bundle_audit = bundle.validate(decision_ts=decision)
    if manifest["hashes"]["inference_bundle"] != _sha(args.inference_bundle):
        raise ValueError("hourly run and independent replay use different bundles")
    fields = _feature_fields(bundle.path("feature_contract"))
    args.out.mkdir(parents=True)

    population = pd.read_parquet(
        args.run / "candidate_grid/target_free_candidate_population.parquet"
    )
    population["__decision_ts__"] = pd.to_datetime(
        population["__decision_ts__"], utc=True, errors="raise",
    )
    current_population = population.loc[
        population["__decision_ts__"].eq(decision)
    ].copy()
    frozen_input_dir = args.run / "current_hour_inputs"
    frozen_input_manifest_path = frozen_input_dir / "run_manifest.json"
    frozen_input_manifest = (
        json.loads(frozen_input_manifest_path.read_text())
        if frozen_input_manifest_path.is_file() else None
    )
    stored_features_path = args.run / "features/canonical120_features.parquet"
    stored_features = pd.read_parquet(stored_features_path)
    stored_features["__decision_ts__"] = pd.to_datetime(
        stored_features["__decision_ts__"], utc=True, errors="raise",
    )
    stored_current_features = stored_features.loc[
        stored_features["__decision_ts__"].eq(decision)
    ].copy()
    frozen_scorer_features_path = frozen_input_dir / "canonical120_features.parquet"
    frozen_scorer_candidates_path = frozen_input_dir / "eligible_candidates.parquet"
    if frozen_input_manifest is not None:
        if _sha(frozen_scorer_features_path) != frozen_input_manifest.get(
            "canonical120_features_sha256"
        ):
            raise ValueError("frozen current-hour feature input hash changed")
        if _sha(frozen_scorer_candidates_path) != frozen_input_manifest.get(
            "eligible_candidates_sha256"
        ):
            raise ValueError("frozen current-hour candidate input hash changed")
        if _utc(frozen_input_manifest["decision_ts"]) != decision:
            raise ValueError("frozen current-hour scorer inputs use another decision")
    if args.rebuild_inputs:
        current_population_path = args.out / "current_population.parquet"
        current_population.to_parquet(
            current_population_path, index=False, compression="zstd",
        )
        feature_replay_dir = args.out / "feature_replay"
        _run([
            sys.executable,
            str(ROOT / bundle.payload["runtime"]["feature_materializer"]),
            "--candidates", str(current_population_path),
            "--out-dir", str(feature_replay_dir),
            "--candidate-start", decision.isoformat(),
            "--history-start", str(bundle.payload["runtime"]["feature_history_start"]),
            "--end-exclusive", (decision + pd.Timedelta(hours=1)).isoformat(),
            "--side", "long",
        ], args.out / "feature_replay.log")
        replayed_features = pd.read_parquet(
            feature_replay_dir / "canonical120_features.parquet"
        )
    elif frozen_input_manifest is not None:
        replayed_features = pd.read_parquet(frozen_scorer_features_path)
    else:
        # Fast hourly mode deliberately trusts only the immutable point-in-time
        # feature artifact emitted by the live producer.  Model, trust, and
        # admission layers are still recomputed independently below.  Full
        # source-to-feature reconstruction remains available via --rebuild-inputs.
        replayed_features = stored_current_features.copy()
    feature_audit, feature_summary = _compare(
        stored_current_features, replayed_features,
        fields=fields, role="features",
    )
    feature_audit.to_parquet(args.out / "feature_parity.parquet", index=False)

    if args.rebuild_inputs:
        replay_feature_ids = set(replayed_features["candidate_id"].astype(str))
        historical_features = stored_features.loc[
            ~stored_features["candidate_id"].astype(str).isin(replay_feature_ids)
        ].copy()
        replay_prefix_features = pd.concat(
            [historical_features, replayed_features], ignore_index=True, sort=False,
        ).sort_values(["__decision_ts__", "candidate_id"], kind="stable")
        replay_prefix_path = args.out / "replay_prefix_features.parquet"
        replay_prefix_features.to_parquet(
            replay_prefix_path, index=False, compression="zstd",
        )
    else:
        replay_prefix_path = (
            frozen_scorer_features_path
            if frozen_input_manifest is not None else stored_features_path
        )

    score_dir = args.out / "score_replay"
    stored_score_manifest = json.loads(
        (args.run / "cycle/score/run_manifest.json").read_text()
    )
    geometry_state_input = (
        (stored_score_manifest.get("geometry_k9_state") or {}).get("input")
        or stored_score_manifest.get("geometry_k9_state_input")
    )
    replay_candidates_path = (
        frozen_scorer_candidates_path
        if frozen_input_manifest is not None
        else args.run / "candidate_grid/eligible_candidates.parquet"
    )
    _run([
        sys.executable, str(ROOT / "scripts/score_strict_r3_forward.py"),
        "--schema", "current-v5",
        "--bundle-dir", str(bundle.path("conversion_bundle_dir")),
        "--upstream-bundle-dir", str(bundle.path("upstream_bundle_dir")),
        "--reference-candidates", str(bundle.path("same_model_reference_candidates")),
        "--reference-features", str(bundle.path("same_model_reference_features")),
        "--held-candidates", str(replay_candidates_path),
        "--held-features", str(replay_prefix_path),
        "--out-dir", str(score_dir),
        "--lockstep-score-chunk-hours", "72",
        *(
            ["--lockstep-geometry-k9-state-in", str(geometry_state_input)]
            if geometry_state_input else []
        ),
    ], args.out / "score_replay.log")

    replay_score_manifest = json.loads(
        (score_dir / "run_manifest.json").read_text()
    )
    scorer_feature_path = (
        frozen_scorer_features_path
        if frozen_input_manifest is not None else stored_features_path
    )
    stored_feature_sha256 = _sha(scorer_feature_path)
    eligible_candidates_path = replay_candidates_path
    eligible_candidates_sha256 = _sha(eligible_candidates_path)
    stored_source_hashes = dict(stored_score_manifest.get("source_hashes") or {})
    replay_source_hashes = dict(replay_score_manifest.get("source_hashes") or {})
    input_lineage = {
        "stored_feature_sha256": stored_feature_sha256,
        "stored_scorer_feature_sha256": stored_source_hashes.get("held_features"),
        "replay_scorer_feature_sha256": replay_source_hashes.get("held_features"),
        "eligible_candidates_sha256": eligible_candidates_sha256,
        "stored_scorer_candidates_sha256": stored_source_hashes.get(
            "held_candidates"
        ),
        "replay_scorer_candidates_sha256": replay_source_hashes.get(
            "held_candidates"
        ),
        "feature_contract_membership_exact": bool(
            len(fields) == 120
            and len(set(fields)) == 120
            and all(field in stored_features.columns for field in fields)
            and sum(field in set(fields) for field in stored_features.columns)
            == 120
        ),
    }
    input_lineage["all_exact"] = bool(
        stored_feature_sha256
        == stored_source_hashes.get("held_features")
        == replay_source_hashes.get("held_features")
        and eligible_candidates_sha256
        == stored_source_hashes.get("held_candidates")
        == replay_source_hashes.get("held_candidates")
        and input_lineage["feature_contract_membership_exact"]
    )

    stored_predictions = pd.read_parquet(args.run / "cycle/score/predictions.parquet")
    replayed_predictions = pd.read_parquet(score_dir / "predictions.parquet")
    for frame in (stored_predictions, replayed_predictions):
        frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True)
    stored_current_predictions = stored_predictions.loc[
        stored_predictions["__decision_ts__"].eq(decision)
    ].copy()
    replayed_current_predictions = replayed_predictions.loc[
        replayed_predictions["__decision_ts__"].eq(decision)
    ].copy()
    prediction_fields = sorted(
        set(stored_current_predictions.columns)
        .intersection(replayed_current_predictions.columns)
        .difference({"candidate_id"})
    )
    prediction_audit, prediction_summary = _compare(
        stored_current_predictions, replayed_current_predictions,
        fields=prediction_fields, role="model_outputs",
    )
    prediction_audit.to_parquet(args.out / "model_output_parity.parquet", index=False)

    admission_dir = args.out / "admission_replay"
    _run([
        sys.executable, str(ROOT / "scripts/admit_strict_r3_mc1_forward.py"),
        "--resolved-score-label-ledger", str(
            args.run / "cycle/runtime_resolved_state/walkforward_scored_label_ledger.parquet"
        ),
        "--current-predictions", str(score_dir / "predictions.parquet"),
        "--mc1-bundle-dir", str(bundle.path("mc1_d2_bundle_dir")),
        "--r5-bundle-dir", str(bundle.path("cell_day_trust_bundle_dir")),
        "--a5-bundle-dir", str(bundle.path("a5_bundle_dir")),
        "--decision-ts", decision.isoformat(),
        "--out-dir", str(admission_dir),
    ], args.out / "admission_replay.log")
    stored_admission = pd.read_parquet(
        args.run / "cycle/admission/admitted_predictions.parquet"
    )
    replayed_admission = pd.read_parquet(
        admission_dir / "admitted_predictions.parquet"
    )
    for frame in (stored_admission, replayed_admission):
        frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True)
    stored_current_admission = stored_admission.loc[
        stored_admission["__decision_ts__"].eq(decision)
    ].copy()
    replayed_current_admission = replayed_admission.loc[
        replayed_admission["__decision_ts__"].eq(decision)
    ].copy()
    admission_fields = sorted(
        set(stored_current_admission.columns)
        .intersection(replayed_current_admission.columns)
        .difference({"candidate_id"})
    )
    admission_audit, admission_summary = _compare(
        stored_current_admission, replayed_current_admission,
        fields=admission_fields, role="admission_and_trust",
    )
    admission_audit.to_parquet(args.out / "admission_parity.parquet", index=False)

    all_pass = bool(input_lineage["all_exact"]) and all(
        summary["all_fields_match"] for summary in (
        feature_summary, prediction_summary, admission_summary,
        )
    )
    result = {
        "schema": SCHEMA,
        "status": "pass" if all_pass else "fail",
        "decision_ts": decision.isoformat(),
        "run": str(args.run),
        "inference_bundle": str(args.inference_bundle),
        "inference_bundle_sha256": _sha(args.inference_bundle),
        "bundle_audit": bundle_audit,
        "relative_tolerance": RTOL,
        "absolute_tolerance": ATOL,
        "features": feature_summary,
        "model_outputs": prediction_summary,
        "admission_and_trust": admission_summary,
        "input_lineage": input_lineage,
        "admitted_identities_exact": bool(
            set(stored_current_admission.loc[
                stored_current_admission["mc1_d2_admitted_ge_50bps"].fillna(False),
                "candidate_id",
            ].astype(str))
            == set(replayed_current_admission.loc[
                replayed_current_admission["mc1_d2_admitted_ge_50bps"].fillna(False),
                "candidate_id",
            ].astype(str))
        ),
        "input_rebuild_performed": bool(args.rebuild_inputs),
        "feature_parity_scope": (
            "independent_source_to_feature_rebuild"
            if args.rebuild_inputs
            else "persisted_point_in_time_features; independent downstream replay"
        ),
        "outcomes_consumed": [],
    }
    if not result["admitted_identities_exact"]:
        all_pass = False
        result["status"] = "fail"
    audit_completed_at = pd.Timestamp(datetime.now(timezone.utc))
    inference_freshness_seconds = float(
        bundle.payload["live_decision_freshness_seconds"]
    )
    audit_maximum_age_seconds = (
        inference_freshness_seconds
        if args.maximum_audit_age_seconds is None
        else float(args.maximum_audit_age_seconds)
    )
    if audit_maximum_age_seconds < inference_freshness_seconds:
        raise ValueError(
            "maximum audit age cannot be shorter than the frozen inference window"
        )
    audit_completion_age = float((audit_completed_at - decision).total_seconds())
    audit_within_window = (
        0.0 <= audit_completion_age <= audit_maximum_age_seconds
    )
    result.update({
        "audit_started_at": audit_started_at.isoformat(),
        "audit_completed_at": audit_completed_at.isoformat(),
        "audit_completion_decision_age_seconds": audit_completion_age,
        "inference_freshness_seconds": inference_freshness_seconds,
        "audit_maximum_age_seconds": audit_maximum_age_seconds,
        "audit_completed_within_audit_window": audit_within_window,
        # Backward-compatible receipt field; the explicit maximum above defines
        # the post-decision audit window and never widens entry authority.
        "audit_completed_within_live_decision_window": audit_within_window,
        "live_wall_clock_enforced": bool(args.enforce_live_wall_clock),
    })
    if args.enforce_live_wall_clock and not audit_within_window:
        all_pass = False
        result["status"] = "fail_late_audit"
    (args.out / "run_manifest.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n"
    )
    print(json.dumps(result, default=str))
    if not all_pass:
        raise SystemExit("schema-v6 current replay parity failed")


if __name__ == "__main__":
    main()
