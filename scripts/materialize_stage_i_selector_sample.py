#!/usr/bin/env python3
"""Materialise one immutable, stratified Stage-I selector matrix."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.config import CFG
from extreme_price_movements.packb_static_point_feature_loader import (
    FrozenFeatureContract,
    _feature_contract_digest,
)
from extreme_price_movements.stage_i_feature_selection import (
    STAGE_I_ACTIVE_CONTRACTS,
    resolve_stage_i_feature_universe,
)
from extreme_price_movements.stage_i_production_data_adapter import (
    MonthlyReferencePartition,
    _validate_exact_feature_frame,
    load_reference_ledgers,
    make_static_pit_feature_loader,
    stratified_selector_sample,
)


IDENTITY = ["candidate_id", "__ts__", "__symbol__"]
EXACT_SELECTOR_MIN_COVERAGE = 0.90
REQUIRED_PRODUCTION_EVALUATION_START = pd.Timestamp("2024-01-01T00:00:00Z")


def _subset_contract(contract: FrozenFeatureContract, fields: list[str]) -> FrozenFeatureContract:
    digest = _feature_contract_digest(
        feature_columns=fields,
        candidate_universe_sha256=contract.candidate_universe_sha256,
        source_schema_sha256=contract.source_schema_sha256,
        raw_allowlist_sha256=contract.raw_allowlist_sha256,
        generator_registry_sha256=contract.generator_registry_sha256,
        store_scan_manifest_sha256=contract.store_scan_manifest_sha256,
        coverage_profile_sha256=contract.coverage_profile_sha256,
        min_exact_key_coverage=contract.min_exact_key_coverage,
        min_non_null_feature_coverage=contract.min_non_null_feature_coverage,
        max_feature_columns=contract.max_feature_columns,
        coverage_admission_rejections=contract.coverage_admission_rejections,
    )
    return replace(
        contract, feature_columns=tuple(fields), feature_contract_sha256=digest
    )


def _block_path(root: Path, index: int, fields: list[str]) -> Path:
    field_hash = hashlib.sha256("\n".join(fields).encode("utf-8")).hexdigest()[:12]
    return root / f"block_{index:04d}_{field_hash}.parquet"


def _identity_matches(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    return left.loc[:, IDENTITY].reset_index(drop=True).equals(
        right.loc[:, IDENTITY].reset_index(drop=True)
    )


def _checkpoint_field_groups(
    candidate_fields: list[str], rejected_fields: list[str], width: int
) -> list[tuple[int, list[str]]]:
    rejected = set(rejected_fields)
    groups: list[tuple[int, list[str]]] = []
    for block_index, start in enumerate(range(0, len(candidate_fields), int(width))):
        active = [
            field
            for field in candidate_fields[start:start + int(width)]
            if field not in rejected
        ]
        if active:
            groups.append((block_index, active))
    return groups


def _load_or_create_feature_block(
    ledger: pd.DataFrame,
    *,
    block_fields: list[str],
    block_path: Path,
    loader,
) -> tuple[pd.DataFrame, bool]:
    if block_path.exists():
        cached = pd.read_parquet(block_path)
        if list(cached.columns) != [*IDENTITY, *block_fields]:
            raise ValueError(f"invalid selector checkpoint schema: {block_path}")
        if not _identity_matches(ledger, cached):
            raise ValueError(f"selector checkpoint identity differs: {block_path}")
        return cached, True
    block = _validate_exact_feature_frame(
        ledger, loader(ledger, block_fields), block_fields
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{block_path.name}.", suffix=".tmp", dir=block_path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        block.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, block_path)
    finally:
        temporary.unlink(missing_ok=True)
    return block, False


def _measure_exact_selector_block(
    ledger: pd.DataFrame,
    loaded: pd.DataFrame,
    fields: list[str],
    *,
    block_index: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Measure the exact selector cohort and retain only eligible columns.

    This is deliberately stricter than relying on the earlier balanced 40K
    discovery sample.  Missing/non-finite values are never filled and the
    threshold is not adjusted to rescue a field.
    """
    expected = [*IDENTITY, *fields]
    if list(loaded.columns) != expected:
        raise ValueError(
            f"selector PIT block {block_index} returned a widened/reordered schema"
        )
    if not _identity_matches(ledger, loaded):
        raise ValueError(f"selector PIT block {block_index} changed exact identities")
    accepted: list[str] = []
    records: list[dict[str, object]] = []
    detail_records: list[dict[str, object]] = []
    normalized = loaded.loc[:, IDENTITY].copy()
    signal = pd.to_datetime(ledger["__ts__"], utc=True, errors="coerce")
    if signal.isna().any():
        raise ValueError("selector cohort has invalid signal timestamps")
    side = ledger["side_name"].astype(str) if "side_name" in ledger else pd.Series("unknown", index=ledger.index)
    source_start = signal.min()
    for feature in fields:
        values = pd.to_numeric(loaded[feature], errors="coerce").to_numpy(dtype=np.float64)
        finite = np.isfinite(values)
        coverage = float(finite.mean()) if len(values) else 0.0
        nonconstant = bool(finite.any() and np.unique(values[finite]).size > 1)
        first_ready = signal.loc[finite].min() if finite.any() else pd.NaT
        if pd.isna(first_ready):
            post_ready = np.zeros(len(values), dtype=bool)
        else:
            post_ready = signal.ge(first_ready).to_numpy()
        evaluation = signal.ge(REQUIRED_PRODUCTION_EVALUATION_START).to_numpy()
        post_rows = int(post_ready.sum())
        post_finite = int((finite & post_ready).sum())
        eval_rows = int(evaluation.sum())
        eval_finite = int((finite & evaluation).sum())
        post_coverage = post_finite / post_rows if post_rows else 0.0
        eval_coverage = eval_finite / eval_rows if eval_rows else 0.0
        prefix_rows = int(signal.lt(first_ready).sum()) if not pd.isna(first_ready) else len(values)
        causal_warmup = bool(prefix_rows and not finite[signal.lt(first_ready).to_numpy()].any())
        if pd.isna(first_ready):
            status = "rejected"
            reason = "never_ready_on_exact_selector_cohort"
        elif first_ready > REQUIRED_PRODUCTION_EVALUATION_START:
            status = "rejected"
            reason = "not_ready_by_required_production_evaluation_start"
        elif post_coverage < EXACT_SELECTOR_MIN_COVERAGE:
            status = "rejected"
            reason = "post_readiness_finite_coverage_below_0.90"
        elif eval_coverage < EXACT_SELECTOR_MIN_COVERAGE:
            status = "rejected"
            reason = "required_evaluation_window_finite_coverage_below_0.90"
        elif not nonconstant:
            status = "rejected"
            reason = "exact_selector_cohort_constant_or_nonincremental"
        else:
            status = "accepted"
            reason = (
                "causal_warmup_prefix_post_readiness_and_evaluation_pass"
                if causal_warmup
                else "exact_selector_cohort_post_readiness_and_evaluation_pass"
            )
            accepted.append(feature)
            normalized_values = values.astype(np.float32)
            # The inferred causal readiness boundary is explicit.  No value is
            # manufactured before it; the prefix remains unavailable/NaN.
            if causal_warmup:
                normalized_values[signal.lt(first_ready).to_numpy()] = np.nan
            normalized[feature] = normalized_values
        records.append({
            "block_index": int(block_index), "feature_name": feature,
            "selector_rows": int(len(values)), "finite_rows": int(finite.sum()),
            "finite_coverage": coverage, "nonconstant": nonconstant,
            "source_history_start_utc": source_start,
            "first_ready_timestamp_utc": first_ready,
            "causal_warmup_prefix": causal_warmup,
            "prefix_rows": prefix_rows,
            "post_readiness_rows": post_rows,
            "post_readiness_finite_rows": post_finite,
            "post_readiness_finite_coverage": post_coverage,
            "post_readiness_missing_rows": post_rows - post_finite,
            "required_evaluation_start_utc": REQUIRED_PRODUCTION_EVALUATION_START,
            "required_evaluation_rows": eval_rows,
            "required_evaluation_finite_rows": eval_finite,
            "required_evaluation_finite_coverage": eval_coverage,
            "threshold": EXACT_SELECTOR_MIN_COVERAGE,
            "status": status, "reason": reason,
            "measurement_scope": "exact_frozen_selector_cohort",
        })
        detail = pd.DataFrame({
            "month": signal.dt.strftime("%Y-%m"), "side_name": side.to_numpy(),
            "finite": finite,
        })
        for (month, side_name), group in detail.groupby(
            ["month", "side_name"], observed=True, sort=True
        ):
            detail_records.append({
                "block_index": int(block_index), "feature_name": feature,
                "month": str(month), "side_name": str(side_name),
                "rows": int(len(group)), "finite_rows": int(group.finite.sum()),
                "finite_coverage": float(group.finite.mean()),
                "hard_gate_applied": False,
                "purpose": "diagnostic_only_month_side_sample_fluctuation",
            })
    return (
        normalized.loc[:, [*IDENTITY, *accepted]],
        pd.DataFrame(records),
        pd.DataFrame(detail_records),
    )


def _contract_with_exact_rejections(
    contract: FrozenFeatureContract,
    *,
    retained_fields: list[str],
    rejection_reasons: dict[str, str],
) -> FrozenFeatureContract:
    combined = dict(contract.coverage_admission_rejections)
    combined.update({str(name): str(reason) for name, reason in rejection_reasons.items()})
    rejections = tuple(sorted(combined.items()))
    digest = _feature_contract_digest(
        feature_columns=retained_fields,
        candidate_universe_sha256=contract.candidate_universe_sha256,
        source_schema_sha256=contract.source_schema_sha256,
        raw_allowlist_sha256=contract.raw_allowlist_sha256,
        generator_registry_sha256=contract.generator_registry_sha256,
        store_scan_manifest_sha256=contract.store_scan_manifest_sha256,
        coverage_profile_sha256=contract.coverage_profile_sha256,
        min_exact_key_coverage=contract.min_exact_key_coverage,
        min_non_null_feature_coverage=EXACT_SELECTOR_MIN_COVERAGE,
        max_feature_columns=contract.max_feature_columns,
        coverage_admission_rejections=rejections,
    )
    return replace(
        contract,
        feature_columns=tuple(retained_fields),
        min_non_null_feature_coverage=EXACT_SELECTOR_MIN_COVERAGE,
        coverage_admission_rejections=rejections,
        feature_contract_sha256=digest,
    )


def _write_coverage_audit(path: Path, audit: pd.DataFrame) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        audit.sort_values(["block_index", "feature_name"], kind="stable").to_parquet(
            temporary, index=False, compression="zstd"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-contract-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selector-rows", type=int, default=80_000)
    parser.add_argument("--feature-columns-per-checkpoint", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--coverage-rejection", action="append", default=[],
        help="Coverage-only field rejection discovered by a full selector pass; repeatable.",
    )
    args = parser.parse_args()
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite selector sample: {args.output_dir}")
    if int(args.feature_columns_per_checkpoint) < 1:
        raise ValueError("--feature-columns-per-checkpoint must be positive")

    manifest = json.loads((args.input_contract_dir / "manifest.json").read_text())
    production_contract = FrozenFeatureContract.from_mapping(json.loads(
        (args.input_contract_dir / "frozen_feature_contract.json").read_text()
    ))
    admitted = set(production_contract.feature_columns)
    candidate_requested = sorted(set(
        name
        for item in STAGE_I_ACTIVE_CONTRACTS
        for name in resolve_stage_i_feature_universe(
            CFG, layer=item.layer, side=item.side, head=item.head
        )
        if name in admitted
    ))
    manual_rejected = sorted(set(map(str, args.coverage_rejection)))
    unknown_rejections = sorted(set(manual_rejected) - set(candidate_requested))
    if unknown_rejections:
        raise ValueError(f"coverage rejection is not in the Stage-I union: {unknown_rejections}")
    partition_frame = pd.read_parquet(args.input_contract_dir / "reference_partitions.parquet")
    partitions = [
        MonthlyReferencePartition(row.path, row.source_month, row.population)
        for row in partition_frame.itertuples(index=False)
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    completed_manifest = args.output_dir / "manifest.json"
    if completed_manifest.exists():
        if args.resume:
            completed = json.loads(completed_manifest.read_text())
            selector_contract_path = args.output_dir / "selector_feature_contract.json"
            exact_audit = completed.get("coverage_audit", {})
            audit_path = args.output_dir / str(exact_audit.get("path", ""))
            if (
                not selector_contract_path.is_file()
                or FrozenFeatureContract.from_mapping(
                    json.loads(selector_contract_path.read_text())
                ).feature_contract_sha256 != completed.get("feature_contract_sha256")
                or not audit_path.is_file()
                or _file_sha256(audit_path) != exact_audit.get("sha256")
            ):
                raise ValueError("completed selector artifact contract/audit checksum drift")
            shutil.rmtree(args.output_dir / "feature_blocks", ignore_errors=True)
            (args.output_dir / "run_state.json").unlink(missing_ok=True)
            print(completed_manifest.read_text(), end="")
            return 0
        raise FileExistsError(f"selector sample is already complete: {args.output_dir}")

    coverage_audit_path = args.output_dir / "selector_exact_feature_coverage_audit.parquet"
    detail_audit_path = args.output_dir / "selector_exact_feature_month_side_coverage.parquet"
    prior_audit = (
        pd.read_parquet(coverage_audit_path)
        if coverage_audit_path.exists() and args.resume
        else pd.DataFrame()
    )
    prior_detail = (
        pd.read_parquet(detail_audit_path)
        if detail_audit_path.exists() and args.resume
        else pd.DataFrame()
    )
    rejection_reasons = {
        str(row.feature_name): str(row.reason)
        for row in prior_audit.loc[prior_audit.status.eq("rejected")].itertuples(index=False)
    } if not prior_audit.empty else {}
    rejection_reasons.update({
        feature: "operator_declared_coverage_rejection"
        for feature in manual_rejected
    })
    rejected = set(rejection_reasons)

    ledger = stratified_selector_sample(
        load_reference_ledgers(partitions),
        max_rows=int(args.selector_rows),
        random_state=42,
    )
    adverse = ledger["t2_tp6_sl4_event"].eq(1.0)
    clear = ledger["robust_clear_event_b25"].eq(1.0)
    ledger["r3_class"] = np.select([adverse, clear], [0, 2], default=1).astype(np.int8)
    ledger["r3_metric_target"] = (
        pd.to_numeric(ledger["robust_clear_soft_b25_t50"], errors="raise")
        - adverse.astype(float)
    ).astype(np.float32)
    ledger_path = args.output_dir / "selector_ledger.parquet"
    if ledger_path.exists():
        prior_ledger = pd.read_parquet(ledger_path)
        if not _identity_matches(ledger, prior_ledger):
            raise ValueError("resume selector cohort differs from the frozen checkpoint")
        ledger = prior_ledger
    else:
        ledger.to_parquet(ledger_path, index=False, compression="zstd")

    checkpoint_root = args.output_dir / "feature_blocks"
    checkpoint_root.mkdir(exist_ok=True)
    block_width = int(args.feature_columns_per_checkpoint)
    blocks: list[tuple[Path, list[str]]] = []
    audit_by_feature = {
        str(row.feature_name): dict(row._asdict())
        for row in prior_audit.itertuples(index=False)
    } if not prior_audit.empty else {}
    detail_frames: list[pd.DataFrame] = [prior_detail] if not prior_detail.empty else []
    for feature in manual_rejected:
        audit_by_feature[feature] = {
            "block_index": -1, "feature_name": feature,
            "selector_rows": int(len(ledger)), "finite_rows": np.nan,
            "finite_coverage": np.nan, "nonconstant": np.nan,
            "source_history_start_utc": ledger["__ts__"].min(),
            "first_ready_timestamp_utc": pd.NaT,
            "causal_warmup_prefix": False, "prefix_rows": np.nan,
            "post_readiness_rows": np.nan, "post_readiness_finite_rows": np.nan,
            "post_readiness_finite_coverage": np.nan,
            "post_readiness_missing_rows": np.nan,
            "required_evaluation_start_utc": REQUIRED_PRODUCTION_EVALUATION_START,
            "required_evaluation_rows": np.nan,
            "required_evaluation_finite_rows": np.nan,
            "required_evaluation_finite_coverage": np.nan,
            "threshold": EXACT_SELECTOR_MIN_COVERAGE,
            "status": "rejected", "reason": "operator_declared_coverage_rejection",
            "measurement_scope": "operator_declared_no_imputation",
        }
    # Form boundaries on the pre-rejection union.  Adding a coverage-only
    # rejection therefore invalidates only its own block rather than shifting
    # and rereading every later feature checkpoint.
    for block_index, start in enumerate(range(0, len(candidate_requested), block_width)):
        original_fields = candidate_requested[start:start + block_width]
        active_fields = [field for field in original_fields if field not in rejected]
        if not active_fields:
            continue
        expected_path = _block_path(checkpoint_root, block_index, active_fields)
        if expected_path.exists():
            loaded = pd.read_parquet(expected_path)
            if list(loaded.columns) != [*IDENTITY, *active_fields] or not _identity_matches(ledger, loaded):
                raise ValueError(f"invalid selector checkpoint: {expected_path}")
        else:
            block_contract = _subset_contract(production_contract, active_fields)
            loader = make_static_pit_feature_loader(
                feature_store_dir=manifest["feature_store"],
                feature_contract=block_contract,
                max_rows_per_batch=4_000,
                max_columns_per_read=min(64, block_width),
            )
            loaded = loader(ledger, active_fields)
        measured, block_audit, block_detail = _measure_exact_selector_block(
            ledger, loaded, active_fields, block_index=block_index
        )
        for row in block_audit.itertuples(index=False):
            audit_by_feature[str(row.feature_name)] = dict(row._asdict())
            if row.status == "rejected":
                rejected.add(str(row.feature_name))
                rejection_reasons[str(row.feature_name)] = str(row.reason)
        if not block_detail.empty:
            # Replace stale diagnostics for every field remeasured in this block.
            if detail_frames:
                prior_combined = pd.concat(detail_frames, ignore_index=True)
                prior_combined = prior_combined.loc[
                    ~prior_combined.feature_name.astype(str).isin(active_fields)
                ]
                detail_frames = [prior_combined]
            detail_frames.append(block_detail)
        accepted_fields = [
            field for field in active_fields
            if audit_by_feature[field]["status"] == "accepted"
        ]
        if accepted_fields:
            block_path = _block_path(checkpoint_root, block_index, accepted_fields)
            if block_path.exists():
                cached = pd.read_parquet(block_path)
                if list(cached.columns) != [*IDENTITY, *accepted_fields] or not _identity_matches(ledger, cached):
                    raise ValueError(f"invalid selector checkpoint: {block_path}")
            else:
                accepted_block = measured.loc[:, [*IDENTITY, *accepted_fields]]
                descriptor, temporary_name = tempfile.mkstemp(
                    prefix=f".{block_path.name}.", suffix=".tmp", dir=checkpoint_root
                )
                os.close(descriptor)
                temporary = Path(temporary_name)
                try:
                    accepted_block.to_parquet(temporary, index=False, compression="zstd")
                    os.replace(temporary, block_path)
                finally:
                    temporary.unlink(missing_ok=True)
            blocks.append((block_path, accepted_fields))
        current_audit = pd.DataFrame(list(audit_by_feature.values()))
        _write_coverage_audit(coverage_audit_path, current_audit)
        if detail_frames:
            _write_coverage_audit(
                detail_audit_path, pd.concat(detail_frames, ignore_index=True)
            )
        state = {
            "schema": "stage_i_selector_checkpoint_v1",
            "status": "in_progress",
            "production_input_feature_contract_sha256": production_contract.feature_contract_sha256,
            "rows": int(len(ledger)),
            "completed_blocks": int(sum(path.exists() for path, _ in blocks)),
            "latest_original_block_index": int(block_index),
            "retained_features_so_far": int(sum(len(fields) for _, fields in blocks)),
            "rejected_features_so_far": sorted(rejected),
        }
        (args.output_dir / "run_state.json").write_text(json.dumps(state, indent=2) + "\n")

    retained_fields = [field for field in candidate_requested if field not in rejected]
    contract = _contract_with_exact_rejections(
        production_contract,
        retained_fields=retained_fields,
        rejection_reasons=rejection_reasons,
    )
    (args.output_dir / "selector_feature_contract.json").write_text(
        json.dumps(contract.to_dict(), indent=2) + "\n", encoding="utf-8"
    )

    feature_frame = ledger.loc[:, IDENTITY].copy()
    for block_path, block_fields in blocks:
        block = pd.read_parquet(block_path)
        if not _identity_matches(feature_frame, block):
            raise ValueError(f"selector checkpoint identity differs at finalization: {block_path}")
        feature_frame.loc[:, block_fields] = block.loc[:, block_fields].to_numpy()
    if list(feature_frame.columns) != [*IDENTITY, *retained_fields]:
        raise ValueError("final selector matrix differs from revised exact feature contract")
    feature_frame.to_parquet(
        args.output_dir / "selector_features.parquet", index=False, compression="zstd"
    )
    population_summary = (
        ledger.groupby(
            ["population_segment", "source_month", "side_name"],
            observed=True,
        )
        .size().rename("rows").reset_index()
    )
    population_summary.to_parquet(
        args.output_dir / "population_summary.parquet", index=False, compression="zstd"
    )
    final_audit = pd.DataFrame(list(audit_by_feature.values())).sort_values(
        ["block_index", "feature_name"], kind="stable"
    )
    _write_coverage_audit(coverage_audit_path, final_audit)
    final_detail = (
        pd.concat(detail_frames, ignore_index=True).sort_values(
            ["block_index", "feature_name", "month", "side_name"], kind="stable"
        )
        if detail_frames else pd.DataFrame()
    )
    if not final_detail.empty:
        _write_coverage_audit(detail_audit_path, final_detail)
    warmup = final_audit.loc[
        final_audit.status.eq("accepted")
        & final_audit.causal_warmup_prefix.astype(bool)
    ]
    out = {
        "schema": "stage_i_selector_sample_v1",
        "status": "complete",
        "rows": int(len(ledger)),
        "long_rows": int(ledger.side_name.eq("long").sum()),
        "short_rows": int(ledger.side_name.eq("short").sum()),
        "feature_columns": int(len(contract.feature_columns)),
        "feature_contract_sha256": contract.feature_contract_sha256,
        "production_input_feature_contract_sha256": production_contract.feature_contract_sha256,
        "coverage_only_rejections": sorted(rejected),
        "coverage_rejection_reasons": {
            feature: rejection_reasons[feature] for feature in sorted(rejected)
        },
        "exact_selector_coverage_contract": {
            "threshold": EXACT_SELECTOR_MIN_COVERAGE,
            "overall_coverage_is_not_the_gate_when_a_causal_warmup_prefix_exists": True,
            "post_readiness_aggregate_gate": EXACT_SELECTOR_MIN_COVERAGE,
            "required_evaluation_start_utc": REQUIRED_PRODUCTION_EVALUATION_START.isoformat(),
            "required_evaluation_aggregate_gate": EXACT_SELECTOR_MIN_COVERAGE,
            "month_side_rows_are_diagnostic_not_hard_gates": True,
            "no_imputation": True,
        },
        "causal_warmup_prefix_features": {
            str(row.feature_name): {
                "source_history_start_utc": pd.Timestamp(row.source_history_start_utc).isoformat(),
                "first_ready_timestamp_utc": pd.Timestamp(row.first_ready_timestamp_utc).isoformat(),
                "prefix_rows": int(row.prefix_rows),
                "post_readiness_finite_coverage": float(row.post_readiness_finite_coverage),
                "required_evaluation_finite_coverage": float(row.required_evaluation_finite_coverage),
                "reason": str(row.reason),
            }
            for row in warmup.itertuples(index=False)
        },
        "coverage_audit": {
            "path": coverage_audit_path.name,
            "sha256": _file_sha256(coverage_audit_path),
            "month_side_path": detail_audit_path.name if detail_audit_path.exists() else None,
            "month_side_sha256": _file_sha256(detail_audit_path) if detail_audit_path.exists() else None,
        },
        "min_signal_ts": ledger["__ts__"].min().isoformat(),
        "max_signal_ts": ledger["__ts__"].max().isoformat(),
        "r3_class_counts": {
            str(int(key)): int(value) for key, value in ledger.r3_class.value_counts().sort_index().items()
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(out, indent=2) + "\n")
    # Checkpoints are useful only until the immutable combined selector is
    # committed.  Remove the duplicate blocks after the manifest is durable.
    shutil.rmtree(checkpoint_root)
    (args.output_dir / "run_state.json").unlink(missing_ok=True)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
