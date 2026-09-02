#!/usr/bin/env python3
"""Fail-closed wiring audit for the strict-R3 long-only lock-step pipeline.

The original schema-v2 audit understood the retired ``month=*`` bundle layout.
The executable producer uses immutable four-week ``cutoff=*`` bundles and a
target-free monthly source store.  This audit validates that actual contract
directly instead of silently treating the repaired layout as legacy input.
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

from extreme_price_movements.strict_r3_canonical_v2 import (  # noqa: E402
    GEOMETRY_END,
    GEOMETRY_START,
    require_single_geometry_hash,
)
from extreme_price_movements.strict_r3_canonical_current import (  # noqa: E402
    load_monthly_upstream_bundle,
)


def _declared_reserve_days(manifest: dict[str, Any]) -> int:
    """Return one internally consistent, positive reserve-window contract."""
    candidates = [
        manifest.get("reference_window_days"),
        manifest.get("calibration_reserve", {}).get("days"),
        manifest.get("prequential_ledger", {}).get("reference_window_days"),
    ]
    declared = {int(value) for value in candidates if value is not None}
    if len(declared) != 1:
        raise ValueError(f"inconsistent reserve-window declarations: {sorted(declared)}")
    days = declared.pop()
    if days <= 0:
        raise ValueError("reserve-window days must be positive")
    return days


def _same_model_reference_name(reserve_days: int) -> str:
    return f"same_conversion_model_prior{reserve_days}"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--walkforward-dir", type=Path, required=True)
    parser.add_argument("--prequential-ledger", type=Path, required=True)
    parser.add_argument("--source-panel", type=Path, required=True)
    parser.add_argument("--geometry-bundle", type=Path, required=True)
    parser.add_argument("--portfolio-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def _sha(path: Path) -> str:
    if path.is_dir():
        return _sha(path / "run_manifest.json")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _target_free_source_audit(
    source: Path,
    fields: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Audit frozen source coverage in bounded month-sized reads."""
    if not source.is_dir():
        partitions = [source]
        detail: dict[str, Any] = {"kind": "single_parquet", "sha256": _sha(source)}
    else:
        manifest_path = source / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema") != "strict_r3_targetfree_month_store_v1":
            raise ValueError("source directory is not the strict target-free monthly store")
        declared = {str(value) for value in manifest.get("fields", [])}
        if not set(fields).issubset(declared):
            raise ValueError("source store lacks a frozen base-contract feature")
        partitions = sorted(source.glob("month=*"))
        detail = {
            "kind": "monthly_target_free_store",
            "manifest_sha256": _sha(manifest_path),
            "months": int(len(partitions)),
        }
    if not partitions:
        raise ValueError("target-free source store has no month partitions")
    counts = {field: 0 for field in fields}
    uniques: dict[str, set[float]] = {field: set() for field in fields}
    rows = 0
    non_long_rows = 0
    duplicate_rows = 0
    for path in partitions:
        frame = pd.read_parquet(path, columns=["candidate_id", "side_name", *fields])
        rows += len(frame)
        non_long_rows += int((~frame["side_name"].astype(str).str.lower().eq("long")).sum())
        duplicate_rows += int(frame["candidate_id"].duplicated().sum())
        for field in fields:
            value = pd.to_numeric(frame[field], errors="coerce").replace([np.inf, -np.inf], np.nan)
            finite = value.dropna()
            counts[field] += int(len(finite))
            # We only need to prove non-constancy: cap the retained set at two
            # values so the audit remains bounded even for 120 high-cardinality
            # fields across multi-year stores.
            if len(uniques[field]) < 2:
                uniques[field].update(float(item) for item in finite.iloc[:1000].unique())
                if len(uniques[field]) > 2:
                    uniques[field] = set(list(uniques[field])[:2])
    coverage = [
        {
            "field": field,
            "finite_fraction": float(counts[field] / rows) if rows else np.nan,
            "n_unique_capped": int(len(uniques[field])),
        }
        for field in fields
    ]
    detail.update({
        "rows": int(rows),
        "non_long_rows": int(non_long_rows),
        "within_partition_duplicate_rows": int(duplicate_rows),
    })
    return coverage, detail


def main() -> None:
    args = _args()
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(condition), "detail": detail})

    geometry_manifest = json.loads((args.geometry_bundle / "run_manifest.json").read_text())
    geometry_parent_hash = str(geometry_manifest["bundle_sha256"])
    check(
        "geometry_definition_window",
        geometry_manifest.get("definition_start") == GEOMETRY_START.isoformat()
        and geometry_manifest.get("definition_end_exclusive") == GEOMETRY_END.isoformat(),
        {"start": geometry_manifest.get("definition_start"), "end_exclusive": geometry_manifest.get("definition_end_exclusive")},
    )
    month_rows = geometry_manifest.get("month_rows", {})
    check(
        "geometry_uses_all_three_months",
        all(int(month_rows.get(month, 0)) > 0 for month in ("2024-10", "2024-11", "2024-12")),
        month_rows,
    )

    root_manifest = json.loads((args.walkforward_dir / "run_manifest.json").read_text())
    declared_reserve_days = _declared_reserve_days(root_manifest)
    check(
        "root_reserve_window_contract_is_consistent",
        declared_reserve_days > 0,
        declared_reserve_days,
    )
    predictions = pd.read_parquet(args.walkforward_dir / "walkforward_predictions.parquet")
    prediction_geometry = require_single_geometry_hash(predictions)
    check("predictions_are_long_only", predictions["side_name"].astype(str).str.lower().eq("long").all(), int(len(predictions)))
    forbidden = [
        column for column in predictions.columns
        if any(token in column.lower() for token in ("label", "outcome", "future", "target", "gross_bps", "net_bps"))
    ]
    check("prediction_surface_is_target_free", not forbidden, forbidden)
    check("prediction_identities_unique", not predictions["candidate_id"].duplicated().any(), int(len(predictions)))
    check("root_consumed_no_outcomes_during_scoring", root_manifest.get("outcomes_consumed_during_scoring") == [], root_manifest.get("outcomes_consumed_during_scoring"))
    check("root_has_no_held_percentiles", int(root_manifest.get("held_percentile_operations", -1)) == 0, root_manifest.get("held_percentile_operations"))
    check("root_frozen_geometry_cadence", root_manifest.get("geometry", {}).get("refit_cadence") == "never", root_manifest.get("geometry"))

    bundle_dirs = sorted((args.walkforward_dir / "bundles").glob("cutoff=*"))
    bundle_rows: list[dict[str, Any]] = []
    ordered_base_fields: tuple[str, ...] | None = None
    effective_hashes: set[str] = set()
    parent_hashes: set[str] = set()
    for directory in bundle_dirs:
        upstream_manifest = json.loads((directory / "upstream" / "run_manifest.json").read_text())
        conversion_manifest = json.loads((directory / "conversion" / "run_manifest.json").read_text())
        upstream = load_monthly_upstream_bundle(directory / "upstream")
        base_fields = tuple(str(value) for value in upstream.base_fields)
        heads = tuple(upstream.conditional_heads)
        if ordered_base_fields is None:
            ordered_base_fields = base_fields
        elif base_fields != ordered_base_fields:
            raise AssertionError("four-week bundles changed the ordered base feature contract")
        effective_hashes.add(str(conversion_manifest.get("geometry_bundle_sha256")))
        parent_hashes.add(str(conversion_manifest.get("geometry_parent_bundle_sha256")))
        upstream_reserve_days = int(upstream_manifest.get("calibration_reserve_days", -1))
        conversion_reserve_days = int(conversion_manifest.get("calibration_reserve_days", -1))
        reserve_days_match = (
            upstream_reserve_days == declared_reserve_days
            and conversion_reserve_days == declared_reserve_days
        )
        reserve_start_match = (
            upstream_manifest.get("calibration_reserve_start")
            == conversion_manifest.get("calibration_reserve_start")
        )
        reserve_scope_match = (
            "excluded from all upstream supervised" in str(upstream_manifest.get("calibration_reserve_contract", ""))
            and "excluded from all conversion supervised" in str(conversion_manifest.get("calibration_reserve_contract", ""))
        )
        bundle_rows.append({
            "cutoff": directory.name.split("=", 1)[-1],
            "base_fields": len(base_fields),
            "residual_heads": len(heads),
            "geometry_effective_hash": conversion_manifest.get("geometry_bundle_sha256"),
            "geometry_parent_hash": conversion_manifest.get("geometry_parent_bundle_sha256"),
            "geometry_refit_cadence": conversion_manifest.get("geometry_refit_cadence"),
            "residual_target": upstream_manifest.get("residual_target"),
            "severe_target": conversion_manifest.get("severe_target"),
            "reserve_days_match": reserve_days_match,
            "declared_reserve_days": declared_reserve_days,
            "upstream_reserve_days": upstream_reserve_days,
            "conversion_reserve_days": conversion_reserve_days,
            "reserve_start_match": reserve_start_match,
            "reserve_scope_match": reserve_scope_match,
        })
    check("lockstep_bundles_exist", bool(bundle_rows), len(bundle_rows))
    check("geometry_never_refit_per_block", all(row["geometry_refit_cadence"] == "never" for row in bundle_rows), bundle_rows)
    check("geometry_parent_is_one_frozen_oct_dec_bundle", parent_hashes == {geometry_parent_hash}, sorted(parent_hashes))
    check("one_effective_geometry_contract", len(effective_hashes) == 1 and prediction_geometry in effective_hashes, sorted(effective_hashes))
    check("lockstep_feature_and_head_contract", all(row["base_fields"] == 120 and row["residual_heads"] == 10 for row in bundle_rows), bundle_rows)
    check("residuals_use_selected_policy_net", all("selected-policy net" in str(row["residual_target"]).lower() for row in bundle_rows), sorted(set(str(row["residual_target"]) for row in bundle_rows)))
    check("declared_reserve_excluded_from_all_active_fits", all(
        row["reserve_days_match"] and row["reserve_start_match"] and row["reserve_scope_match"]
        for row in bundle_rows
    ), bundle_rows)

    if ordered_base_fields is None:
        raise ValueError("no lock-step bundle is available for source-feature audit")
    coverage_rows, source_detail = _target_free_source_audit(args.source_panel, list(ordered_base_fields))
    check("source_model_fields_each_cover_at_least_90pct", all(row["finite_fraction"] >= .90 for row in coverage_rows), sorted(coverage_rows, key=lambda row: row["finite_fraction"])[:10])
    check("source_model_fields_all_vary", all(row["n_unique_capped"] > 1 for row in coverage_rows), [row for row in coverage_rows if row["n_unique_capped"] <= 1])
    check("source_panel_long_only", int(source_detail["non_long_rows"]) == 0, source_detail)
    check("source_partition_identities_unique", int(source_detail["within_partition_duplicate_rows"]) == 0, source_detail)

    conversion_audit = pd.read_parquet(args.walkforward_dir / "conversion_reference_audit.parquet")
    check("same_bundle_for_held_and_reference", conversion_audit["same_conversion_model_reference_and_held"].fillna(False).all() and conversion_audit["same_upstream_bundle_reference_and_held"].fillna(False).all(), int(len(conversion_audit)))
    check("all_upstream_scores_prequential", conversion_audit["upstream_scores_are_prequential_lockstep"].fillna(False).all(), int(len(conversion_audit)))
    check("no_held_window_percentiles", pd.to_numeric(conversion_audit["held_percentile_operations"], errors="coerce").eq(0).all(), conversion_audit["held_percentile_operations"].tolist())
    check("conversion_geometry_never_refit", conversion_audit["geometry_refit_cadence"].eq("never").all() and set(conversion_audit["geometry_parent_bundle_sha256"].astype(str)) == {geometry_parent_hash}, int(len(conversion_audit)))
    expected_reference = _same_model_reference_name(declared_reserve_days)
    check(
        "final_score_uses_same_model_declared_prior_window",
        conversion_audit["final_reference"].eq(expected_reference).all(),
        {
            "expected": expected_reference,
            "observed": conversion_audit["final_reference"].unique().tolist(),
        },
    )

    block_audit = pd.read_parquet(args.walkforward_dir / "lockstep_block_audit.parquet")
    cutoff = pd.to_datetime(block_audit["cutoff"], utc=True)
    reserve_start = pd.to_datetime(block_audit["reserve_start"], utc=True)
    expected_reserve_start = cutoff - pd.to_timedelta(declared_reserve_days, unit="D")
    reserve_days = pd.to_numeric(block_audit["reserve_days"], errors="coerce")
    check(
        "each_declared_reserve_precedes_activation",
        reserve_start.lt(cutoff).all()
        and reserve_days.eq(declared_reserve_days).all()
        and reserve_start.eq(expected_reserve_start).all(),
        {
            "rows": int(len(block_audit)),
            "declared_reserve_days": declared_reserve_days,
            "start_mismatches": int((~reserve_start.eq(expected_reserve_start)).sum()),
        },
    )
    check("same_refit_cutoff", block_audit["same_refit_cutoff"].fillna(False).all() and block_audit["full_shared_reserve"].fillna(False).all(), int(len(block_audit)))

    ledger = pd.read_parquet(args.prequential_ledger, columns=["candidate_id", "stack_is_prequential", "policy_path_valid", "policy_gross_bps", "policy_net_bps", "policy_cost_bps"])
    check("ledger_identities_unique", not ledger["candidate_id"].duplicated().any(), int(len(ledger)))
    check("all_downstream_ledger_rows_prequential", ledger["stack_is_prequential"].fillna(False).all(), int(len(ledger)))
    valid = ledger["policy_path_valid"].fillna(False).astype(bool)
    cost = pd.to_numeric(ledger.loc[valid, "policy_gross_bps"], errors="coerce") - pd.to_numeric(ledger.loc[valid, "policy_net_bps"], errors="coerce")
    check("policy_cost_applied_once", len(cost) > 0 and np.allclose(cost, 100.0, atol=1e-4), {"rows": int(len(cost)), "declared": sorted(pd.to_numeric(ledger.loc[valid, "policy_cost_bps"], errors="coerce").dropna().unique().tolist())})

    if args.portfolio_dir is not None:
        manifest = json.loads((args.portfolio_dir / "run_manifest.json").read_text())
        check("portfolio_is_global_auction", "global auction" in str(manifest.get("portfolio", "")), manifest.get("portfolio"))
        check("portfolio_margin_cap_80pct", "80% margin cap" in str(manifest.get("portfolio", "")), manifest.get("portfolio"))
        decisions = pd.read_parquet(args.portfolio_dir / "portfolio_decisions.parquet")
        accepted = decisions.loc[decisions["accepted"].fillna(False).astype(bool)].copy()
        check("portfolio_observed_concurrency_never_exceeds_8", pd.to_numeric(decisions["open_positions_after"], errors="coerce").max() <= 8, int(len(accepted)))
        accepted_by_bar = accepted.groupby(pd.to_datetime(accepted["timestamp"], utc=True), sort=False).size()
        check("portfolio_observed_new_entries_never_exceed_2_per_bar", int(accepted_by_bar.max()) <= 2 if len(accepted_by_bar) else True, int(accepted_by_bar.max()) if len(accepted_by_bar) else 0)

    failures = [value for value in checks if not value["passed"]]
    report = {
        "schema": "strict_r3_lockstep_long_wiring_audit_v2",
        "passed": not failures,
        "checks": checks,
        "failures": failures,
        "source": source_detail,
        "geometry_parent_bundle_sha256": geometry_parent_hash,
        "geometry_effective_hashes": sorted(effective_hashes),
        "walkforward_rows": int(len(predictions)),
        "lockstep_bundles": int(len(bundle_rows)),
        "declared_reserve_days": declared_reserve_days,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({"event": "complete", **report}, default=str))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
