#!/usr/bin/env python3
"""Fit and score canonical R5 independently at every producer cutoff.

The input score ledger must already be strict-prequential.  The supplied
Cell-day sidecar must be the causal 28-calendar-day, equal-day, symmetric-15%
trim map.  For each held producer block this runner:

1. fits R5 only on earlier rows whose policy labels resolved before cutoff;
2. persists an immutable cutoff-local bundle;
3. scores the complete held population without outcomes; and
4. fails closed, while preserving candidate identity, when support is absent.

No bundle or prediction is shared across cutoffs.  Geometry/K9 is not fitted
here: every row must come from one frozen geometry bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.strict_r3_cell_day_trust import (  # noqa: E402
    POSTERIOR_CONTRACT_PATH,
    persist_cell_day_residual_trust_bundle,
    train_cell_day_residual_trust_bundle,
)


SCHEMA = "strict_r3_cell_day_residual_trust_walkforward_v1"
CANONICAL_MAP_FIELD = "cell_day_trim_15pct__expected_net_bps"
CANONICAL_ADMITTED_FIELD = "cell_day_trim_15pct__admitted"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="raise")


def _has_full_training_window(
    mapped_history_start: object,
    *,
    required_start: pd.Timestamp,
) -> bool:
    """Return whether the full declared nine-month mapped history exists."""
    if pd.isna(mapped_history_start):
        return False
    return pd.Timestamp(mapped_history_start) <= required_start


def _load_map_contract(path: Path, *, expected_field: str) -> dict[str, Any]:
    manifest_path = Path(path).parent / "run_manifest.json"
    if not manifest_path.exists():
        raise ValueError("Cell-day provenance requires its sibling run_manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "strict_r3_cell_day_bayesian_ev_map_ablation_v1":
        raise ValueError("Cell-day provenance has an unknown schema")
    if int(manifest.get("rolling_window_days", -1)) != 28:
        raise ValueError("canonical R5 requires a physical 28-calendar-day map")
    if expected_field != CANONICAL_MAP_FIELD:
        raise ValueError(f"canonical R5 requires {CANONICAL_MAP_FIELD}")
    if "cell_day_trim_15pct" not in set(manifest.get("arms", ())):
        raise ValueError("Cell-day provenance lacks the symmetric 15% trim arm")
    weighting = str(manifest.get("period_weighting", ""))
    if "one observation per UTC day" not in weighting:
        raise ValueError("canonical R5 requires equal Cell-day weighting")
    return manifest


def _join_inputs(
    score: pd.DataFrame,
    mapped: pd.DataFrame,
    *,
    expected_field: str,
    admitted_field: str,
) -> pd.DataFrame:
    for frame, name in ((score, "score"), (mapped, "Cell-day")):
        if frame["candidate_id"].duplicated().any():
            raise ValueError(f"{name} ledger contains duplicate candidate IDs")
    required_score = {
        "candidate_id", "__decision_ts__", "calibration_activation_ts",
        "policy_label_available_ts", "policy_path_valid", "policy_net_bps",
        "geometry_bundle_sha256", "stack_is_prequential", "final_score",
    }
    required_map = {
        "candidate_id", "__decision_ts__", expected_field, admitted_field,
    }
    missing_score = sorted(required_score.difference(score.columns))
    missing_map = sorted(required_map.difference(mapped.columns))
    if missing_score:
        raise ValueError(f"score ledger lacks: {missing_score}")
    if missing_map:
        raise ValueError(f"Cell-day provenance lacks: {missing_map}")
    score = score.copy()
    mapped = mapped.loc[:, sorted(required_map)].copy().rename(columns={
        "__decision_ts__": "__map_decision_ts__",
        expected_field: "raw_expected_bps",
        admitted_field: "raw_cell_day_admitted",
    })
    score["candidate_id"] = score["candidate_id"].astype(str)
    mapped["candidate_id"] = mapped["candidate_id"].astype(str)
    score["__decision_ts__"] = _utc(score["__decision_ts__"])
    mapped["__map_decision_ts__"] = _utc(mapped["__map_decision_ts__"])
    score["calibration_activation_ts"] = _utc(score["calibration_activation_ts"])
    score["policy_label_available_ts"] = pd.to_datetime(
        score["policy_label_available_ts"], utc=True, errors="coerce",
    )
    joined = score.merge(mapped, on="candidate_id", how="left", validate="one_to_one")
    overlap = joined["__map_decision_ts__"].notna()
    if not joined.loc[overlap, "__decision_ts__"].eq(
        joined.loc[overlap, "__map_decision_ts__"]
    ).all():
        raise ValueError("Cell-day provenance identity/timestamp mismatch")
    if not joined["stack_is_prequential"].fillna(False).astype(bool).all():
        raise ValueError("R5 source contains non-prequential upstream rows")
    geometry = joined["geometry_bundle_sha256"].dropna().astype(str).unique()
    if len(geometry) != 1:
        raise ValueError("R5 walk-forward requires one frozen geometry/K9 bundle")
    return joined


def run_walkforward(
    *,
    scored_ledger: Path,
    cell_day_provenance: Path,
    out_dir: Path,
    evaluation_start: object,
    evaluation_end: object,
    integration_contract: Path = POSTERIOR_CONTRACT_PATH,
    expected_field: str = CANONICAL_MAP_FIELD,
    admitted_field: str = CANONICAL_ADMITTED_FIELD,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"immutable R5 walk-forward output exists: {out_dir}")
    map_manifest = _load_map_contract(cell_day_provenance, expected_field=expected_field)
    score = pd.read_parquet(scored_ledger)
    mapped = pd.read_parquet(cell_day_provenance)
    joined = _join_inputs(
        score, mapped, expected_field=expected_field, admitted_field=admitted_field,
    )
    scored_ledger_sha256 = _sha(scored_ledger)
    cell_day_provenance_sha256 = _sha(cell_day_provenance)
    cell_day_manifest_path = Path(cell_day_provenance).parent / "run_manifest.json"
    cell_day_manifest_sha256 = _sha(cell_day_manifest_path)
    start = pd.Timestamp(evaluation_start)
    end = pd.Timestamp(evaluation_end)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    evaluation = joined.loc[
        joined["__decision_ts__"].ge(start) & joined["__decision_ts__"].lt(end)
    ].copy()
    if evaluation.empty:
        raise ValueError("R5 evaluation window is empty")
    if evaluation["calibration_activation_ts"].isna().any():
        raise ValueError("R5 held rows lack producer activation timestamps")

    out_dir.mkdir(parents=True)
    bundle_root = out_dir / "bundles"
    bundle_root.mkdir()
    prediction_parts: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    mapped_history_start = joined.loc[
        np.isfinite(pd.to_numeric(joined["raw_expected_bps"], errors="coerce")),
        "__decision_ts__",
    ].min()
    cutoffs = evaluation["calibration_activation_ts"].drop_duplicates().sort_values()
    for cutoff in cutoffs:
        cutoff = pd.Timestamp(cutoff)
        held = evaluation.loc[evaluation["calibration_activation_ts"].eq(cutoff)].copy()
        if held["__decision_ts__"].lt(cutoff).any():
            raise ValueError("producer held rows precede their activation cutoff")
        token = cutoff.strftime("%Y%m%dT%H%M%SZ")
        bundle_dir = bundle_root / f"cutoff={token}"
        status = "fit"
        error: str | None = None
        bundle = None
        persisted_manifest: dict[str, Any] | None = None
        training_start = cutoff - pd.DateOffset(months=9)
        fit_source = joined.loc[
            joined["__decision_ts__"].ge(training_start)
            & joined["__decision_ts__"].lt(cutoff)
        ].copy()
        try:
            if not _has_full_training_window(
                mapped_history_start, required_start=training_start,
            ):
                raise ValueError(
                    "insufficient resolved prior support: canonical R5 requires a "
                    "complete nine-calendar-month mapped-history window"
                )
            bundle = train_cell_day_residual_trust_bundle(
                fit_source,
                cutoff=cutoff,
                integration_contract_path=integration_contract,
                source_hashes={
                    "scored_ledger": {
                        "path": str(scored_ledger), "sha256": scored_ledger_sha256,
                    },
                    "cell_day_provenance": {
                        "path": str(cell_day_provenance),
                        "sha256": cell_day_provenance_sha256,
                    },
                    "cell_day_manifest_sha256": cell_day_manifest_sha256,
                },
            )
            persisted_manifest = persist_cell_day_residual_trust_bundle(bundle, bundle_dir)
        except ValueError as exc:
            if "insufficient resolved prior support" not in str(exc):
                raise
            status = "fail_closed_insufficient_prior_support"
            error = str(exc)

        base = held.loc[:, [
            "candidate_id", "__decision_ts__", "calibration_activation_ts",
            "geometry_bundle_sha256", "final_score", "raw_expected_bps",
            "raw_cell_day_admitted",
        ]].copy()
        if bundle is None:
            base["trust_posterior_expected_bps"] = np.nan
            base["trust_posterior_available"] = False
            base["trust_posterior_admitted_ge_50bps"] = False
        else:
            inputs = held.loc[:, ["candidate_id", *bundle.fields]].copy()
            inputs["raw_expected_bps"] = pd.to_numeric(
                held["raw_expected_bps"], errors="coerce",
            ).to_numpy(float)
            prediction = bundle.score(inputs)
            base = base.merge(prediction, on="candidate_id", how="left", validate="one_to_one")
            posterior = pd.to_numeric(base["trust_posterior_expected_bps"], errors="coerce")
            base["trust_posterior_available"] = np.isfinite(posterior)
            base["trust_posterior_admitted_ge_50bps"] = (
                base["trust_posterior_available"] & posterior.ge(50.0)
            )
        base["r5_fit_status"] = status
        base["r5_bundle_cutoff"] = cutoff
        prediction_parts.append(base)
        resolved = (
            fit_source["policy_label_available_ts"].lt(cutoff)
            & fit_source["policy_path_valid"].fillna(False).astype(bool)
            & np.isfinite(pd.to_numeric(fit_source["policy_net_bps"], errors="coerce"))
            & np.isfinite(pd.to_numeric(fit_source["raw_expected_bps"], errors="coerce"))
        )
        audits.append({
            "cutoff": cutoff,
            "held_rows": int(len(held)),
            "prior_resolved_map_rows": int(resolved.sum()),
            "status": status,
            "error": error,
            "bundle_sha256": (
                None if persisted_manifest is None
                else persisted_manifest.get("bundle_sha256")
            ),
            "training_start": None if bundle is None else bundle.manifest.get("training_start"),
            "required_training_start": training_start,
            "mapped_history_start": mapped_history_start,
            "full_nine_month_history": _has_full_training_window(
                mapped_history_start, required_start=training_start,
            ),
            "training_rows": None if bundle is None else bundle.manifest.get("train_rows"),
            "posterior_available_rows": int(base["trust_posterior_available"].sum()),
            "posterior_admitted_rows": int(base["trust_posterior_admitted_ge_50bps"].sum()),
        })
        print(json.dumps({"event": "r5_fold_complete", **{
            key: (value.isoformat() if isinstance(value, pd.Timestamp) else value)
            for key, value in audits[-1].items()
        }}), flush=True)

    predictions = pd.concat(prediction_parts, ignore_index=True).sort_values(
        ["__decision_ts__", "candidate_id"], kind="stable",
    ).reset_index(drop=True)
    if predictions["candidate_id"].duplicated().any():
        raise AssertionError("R5 held folds overlap candidate identities")
    if set(predictions["candidate_id"]) != set(evaluation["candidate_id"]):
        raise AssertionError("R5 walk-forward changed the evaluation population")
    prediction_path = out_dir / "cell_day_residual_trust_oof_predictions.parquet"
    audit_path = out_dir / "fold_audit.parquet"
    predictions.to_parquet(prediction_path, index=False, compression="zstd")
    pd.DataFrame(audits).to_parquet(audit_path, index=False)
    manifest = {
        "schema": SCHEMA,
        "side": "long",
        "evaluation_start": start.isoformat(),
        "evaluation_end_exclusive": end.isoformat(),
        "rows": int(len(predictions)),
        "folds": int(len(audits)),
        "fit_folds": int(sum(row["status"] == "fit" for row in audits)),
        "fail_closed_folds": int(sum(row["status"] != "fit" for row in audits)),
        "posterior_available_rows": int(predictions["trust_posterior_available"].sum()),
        "posterior_admitted_rows": int(predictions["trust_posterior_admitted_ge_50bps"].sum()),
        "training_window_months": 9,
        "map_window_calendar_days": 28,
        "map_arm": "cell_day_trim_15pct",
        "missing_posterior": "fail_closed",
        "fit_schedule": "one independently fitted R5 bundle per producer activation cutoff",
        "geometry": "one frozen bundle; never fitted by this runner",
        "geometry_bundle_sha256": str(predictions["geometry_bundle_sha256"].iloc[0]),
        "scored_ledger": str(scored_ledger),
        "scored_ledger_sha256": scored_ledger_sha256,
        "cell_day_provenance": str(cell_day_provenance),
        "cell_day_provenance_sha256": cell_day_provenance_sha256,
        "cell_day_manifest_schema": map_manifest["schema"],
        "integration_contract": str(integration_contract),
        "integration_contract_sha256": _sha(integration_contract),
        "predictions": str(prediction_path),
        "predictions_sha256": _sha(prediction_path),
        "outcomes_used_for_held_scoring": False,
        "all_training_labels_resolved_strictly_before_cutoff": True,
        "raw_k9_memberships_used": False,
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored-ledger", type=Path, required=True)
    parser.add_argument("--cell-day-provenance", type=Path, required=True)
    parser.add_argument("--expected-map-field", default=CANONICAL_MAP_FIELD)
    parser.add_argument("--admitted-map-field", default=CANONICAL_ADMITTED_FIELD)
    parser.add_argument("--integration-contract", type=Path, default=POSTERIOR_CONTRACT_PATH)
    parser.add_argument("--evaluation-start", required=True)
    parser.add_argument("--evaluation-end", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = run_walkforward(
        scored_ledger=args.scored_ledger,
        cell_day_provenance=args.cell_day_provenance,
        out_dir=args.out_dir,
        evaluation_start=args.evaluation_start,
        evaluation_end=args.evaluation_end,
        integration_contract=args.integration_contract,
        expected_field=args.expected_map_field,
        admitted_field=args.admitted_map_field,
    )
    print(json.dumps({"event": "complete", "out_dir": str(args.out_dir), **manifest}))


if __name__ == "__main__":
    main()
