#!/usr/bin/env python3
"""Causality and identity audit for immutable O3-v2 research artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SEMANTIC_NONLABEL_COLUMNS = {
    "candidate_id", "__decision_ts__", "__symbol__", "side_name", "semantic_path_valid",
    "semantic_label_available_ts", "semantic_tbm_path_complete", "semantic_policy_neighbourhood_valid",
}
PROHIBITED_SCORE_PREFIXES = ("semantic_", "policy_")


def _fail(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def _score_receipts(root: Path) -> list[tuple[str, Path]]:
    """Find flat or partitioned target-free score receipts."""
    found: list[tuple[str, Path]] = []
    for arm in sorted((root / "target_free_scores").glob("*")):
        for source in sorted(arm.glob("month=*.parquet")):
            found.append((arm.name, source))
        for part in sorted(arm.glob("month=*")):
            source = part / "scores.parquet"
            if source.exists():
                found.append((arm.name, source))
    return found


def _audit_semantics(root: Path, evidence: dict[str, object], failures: list[str]) -> None:
    manifest = json.loads((root / "run_manifest.json").read_text())
    _fail(manifest.get("schema") == "strict_r3_o3v2_semantics_v1", "unexpected semantic schema", failures)
    monthly: list[dict[str, object]] = []
    for part in sorted((root / "parts").glob("month=*")):
        frame = pd.read_parquet(part / "semantics.parquet")
        valid = frame["semantic_path_valid"].fillna(False).astype(bool)
        label_fields = [column for column in frame if column.startswith("semantic_") and column not in SEMANTIC_NONLABEL_COLUMNS]
        invalid_populated = int(frame.loc[~valid, label_fields].notna().any(axis=1).sum())
        duplicate = int(frame["candidate_id"].duplicated().sum())
        monthly.append({"month": part.name.split("=", 1)[1], "rows": int(len(frame)), "valid_fraction": float(valid.mean()), "invalid_populated": invalid_populated, "duplicate_ids": duplicate})
        _fail(float(valid.mean()) >= .90, f"{part.name}: semantic coverage <90%", failures)
        _fail(invalid_populated == 0, f"{part.name}: invalid paths have semantic labels", failures)
        _fail(duplicate == 0, f"{part.name}: duplicate candidate identities", failures)
    evidence["semantic_months"] = monthly


def _audit_target(root: Path, evidence: dict[str, object], failures: list[str]) -> None:
    manifest = json.loads((root / "run_manifest.json").read_text())
    _fail(str(manifest.get("schema", "")).startswith("strict_r3_o3v2_target_funnel_v"), "unexpected target-funnel schema", failures)
    _fail("six full preceding resolved calendar months before reserve" in str(manifest.get("causality", {}).get("fit", "")), "target funnel does not declare a full reserve-relative six-month fit", failures)
    panels: list[dict[str, object]] = []
    for arm, source in _score_receipts(root):
        frame = pd.read_parquet(source)
        prohibited = [column for column in frame if column.startswith(PROHIBITED_SCORE_PREFIXES)]
        duplicate = int(frame["candidate_id"].duplicated().sum())
        panels.append({"arm": arm, "path": str(source), "rows": int(len(frame)), "prohibited_columns": prohibited, "duplicate_ids": duplicate})
        _fail(not prohibited, f"{arm} {source.name}: target-free receipt contains {prohibited}", failures)
        _fail(duplicate == 0, f"{arm} {source.name}: duplicate candidate identities", failures)
    evidence["target_free_panels"] = panels
    audit = pd.read_parquet(root / "target_funnel_audit.parquet")
    _fail((audit["semantic_valid_fraction"] >= .90).all(), "one or more train folds use <90% semantic coverage", failures)
    _fail((audit["train_rows"] > 0).all() and (audit["held_rows"] > 0).all(), "empty strict fold", failures)
    required_audit = {"train_start", "reserve_start"}
    _fail(required_audit.issubset(audit.columns), "target funnel audit lacks reserve-relative training boundaries", failures)
    if required_audit.issubset(audit.columns):
        starts = pd.to_datetime(audit["train_start"], utc=True, errors="coerce")
        reserves = pd.to_datetime(audit["reserve_start"], utc=True, errors="coerce")
        expected = reserves - pd.DateOffset(months=6)
        _fail(starts.equals(expected), "target funnel train start is not six full months before its reserve", failures)
    evidence["training_folds"] = audit.to_dict("records")


def _audit_support(root: Path, evidence: dict[str, object], failures: list[str]) -> None:
    """Verify support labels influence fitting only, never held score receipts."""
    manifest = json.loads((root / "run_manifest.json").read_text())
    _fail(str(manifest.get("schema", "")).startswith("strict_r3_o3v2_support_funnel_v"), "unexpected support-funnel schema", failures)
    _fail("six full preceding resolved calendar months before reserve" in str(manifest.get("causality", {}).get("fit", "")), "support funnel does not declare a full reserve-relative six-month fit", failures)
    panels: list[dict[str, object]] = []
    for arm, source in _score_receipts(root):
        frame = pd.read_parquet(source)
        prohibited = [column for column in frame if column.startswith(PROHIBITED_SCORE_PREFIXES)]
        duplicate = int(frame["candidate_id"].duplicated().sum())
        panels.append({"arm": arm, "path": str(source), "rows": int(len(frame)), "prohibited_columns": prohibited, "duplicate_ids": duplicate})
        _fail(not prohibited, f"{arm} {source.name}: support score receipt contains {prohibited}", failures)
        _fail(duplicate == 0, f"{arm} {source.name}: duplicate candidate identities", failures)
    evidence["support_target_free_panels"] = panels
    audit = pd.read_parquet(root / "support_funnel_audit.parquet")
    _fail((audit["semantic_valid_fraction"] >= .90).all(), "one or more support folds use <90% semantic coverage", failures)
    _fail((audit["train_rows"] > 0).all() and (audit["held_rows"] > 0).all(), "empty strict support fold", failures)
    _fail((audit["weight_min"] >= .25 - 1e-6).all() and (audit["weight_max"] <= 4.0 + 1e-6).all(), "support weight outside declared bounds", failures)
    required_audit = {"train_start", "reserve_start"}
    _fail(required_audit.issubset(audit.columns), "support funnel audit lacks reserve-relative training boundaries", failures)
    if required_audit.issubset(audit.columns):
        starts = pd.to_datetime(audit["train_start"], utc=True, errors="coerce")
        reserves = pd.to_datetime(audit["reserve_start"], utc=True, errors="coerce")
        expected = reserves - pd.DateOffset(months=6)
        _fail(starts.equals(expected), "support funnel train start is not six full months before its reserve", failures)
    evidence["support_training_folds"] = audit.to_dict("records")


def _audit_score_root(root: Path, evidence: dict[str, object], failures: list[str]) -> None:
    """Audit any sealed target-free specialist/adapter score root."""
    manifest = json.loads((root / "run_manifest.json").read_text())
    rows: list[dict[str, object]] = []
    for arm, source in _score_receipts(root):
        frame = pd.read_parquet(source)
        prohibited = [column for column in frame if column.startswith(PROHIBITED_SCORE_PREFIXES)]
        duplicate = int(frame["candidate_id"].duplicated().sum())
        rows.append({"arm": arm, "path": str(source), "rows": int(len(frame)), "prohibited_columns": prohibited, "duplicate_ids": duplicate})
        _fail(not prohibited, f"{source}: target-free receipt contains {prohibited}", failures)
        _fail(duplicate == 0, f"{source}: duplicate candidate identities", failures)
    _fail(bool(rows), f"{root.name}: no target-free score receipts", failures)
    coverage_path = root / "specialist_coverage.parquet"
    if coverage_path.exists():
        coverage = pd.read_parquet(coverage_path)
        _fail((coverage["field_complete_fraction"] >= .90).all(), f"{root.name}: specialist feature coverage <90%", failures)
    evidence.setdefault("extra_target_free_roots", []).append({"root": root.name, "schema": manifest.get("schema"), "receipts": rows})


def _audit_feature_screen(root: Path, evidence: dict[str, object], failures: list[str]) -> None:
    manifest = json.loads((root / "run_manifest.json").read_text())
    _fail(str(manifest.get("schema", "")).startswith("strict_r3_o3v2_feature_screen_v"), "unexpected feature-screen schema", failures)
    _fail("MDA intentionally deferred" in str(manifest.get("scope", "")), "feature screen unexpectedly claims MDA", failures)
    panel_path = root / "target_free_feature_panel_with_f3.parquet"
    panel = pd.read_parquet(panel_path)
    prohibited = [column for column in panel if column.startswith(PROHIBITED_SCORE_PREFIXES)]
    _fail(not prohibited, f"feature panel contains outcome fields: {prohibited}", failures)
    selected = json.loads((root / "selected_features.json").read_text())
    fields = [field for values in selected.values() for field in values]
    coverage = {field: float(panel[field].notna().mean()) for field in fields}
    _fail(all(value >= .90 for value in coverage.values()), "selected feature coverage <90%", failures)
    evidence["feature_screen"] = {"root": root.name, "selected_fields": fields, "coverage": coverage, "portability_receipt": (root / "feature_screen_portability_metrics.parquet").exists()}


def _audit_g3(root: Path, evidence: dict[str, object], failures: list[str]) -> None:
    contract_path = root / "g3_feature_contracts.json"
    trace_path = root / "g3_strict_oof_trace.parquet"
    _fail(contract_path.exists() and trace_path.exists(), f"{root.name}: missing G3 contract/trace", failures)
    if not contract_path.exists() or not trace_path.exists():
        return
    contract = json.loads(contract_path.read_text())
    _fail(
        contract.get("schema") in {"strict_r3_o3v2_greedy_features_v1", "strict_r3_o3v2_greedy_features_v2"},
        f"{root.name}: unexpected G3 schema",
        failures,
    )
    manifest_path = root / "run_manifest.json"
    manifest: dict[str, object] | None = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        _fail(
            manifest.get("schema") in {"strict_r3_o3v2_greedy_features_v1", "strict_r3_o3v2_greedy_features_v2"},
            f"{root.name}: unexpected G3 run-manifest schema",
            failures,
        )
        if manifest.get("schema") == "strict_r3_o3v2_greedy_features_v2":
            slot = manifest.get("physical_slot_selection", {})
            _fail(bool(slot.get("path")), f"{root.name}: v2 G3 lacks physical-slot lineage", failures)
            _fail(
                slot.get("name") in {"cap100_ordinary", "cap80_ordinary", "cap120_equal_month", "cap40_equal_month", "cap60_equal_month"},
                f"{root.name}: v2 G3 has an invalid physical slot",
                failures,
            )
        training = manifest.get("training", {})
        _fail(bool(training.get("full_window_required")), f"{root.name}: G3 manifest does not require full train windows", failures)
        _fail(int(training.get("calendar_months", -1)) == 6, f"{root.name}: G3 calendar-month contract is not six", failures)
        _fail(int(training.get("reserve_days", -1)) == 28, f"{root.name}: G3 reserve contract is not 28 days", failures)
        try:
            history_start = pd.Timestamp(str(manifest["history_start"]))
            months = tuple(pd.Timestamp(f"{token}-01", tz="UTC") for token in manifest["development_months"])
            required_start = min(
                (month - pd.Timedelta(days=28)) - pd.DateOffset(months=6)
                for month in months
            )
        except (KeyError, TypeError, ValueError):
            failures.append(f"{root.name}: G3 full-window provenance is malformed")
        else:
            _fail(history_start <= required_start, f"{root.name}: G3 history begins after a required training window", failures)
    trace = pd.read_parquet(trace_path)
    _fail(bool(len(trace)), f"{root.name}: empty G3 trace", failures)
    rows: list[dict[str, object]] = []
    for source in sorted((root / "target_free_scores").glob("*/*.parquet")):
        frame = pd.read_parquet(source)
        prohibited = [column for column in frame if column.startswith(PROHIBITED_SCORE_PREFIXES)]
        rows.append({"path": str(source), "rows": int(len(frame)), "prohibited_columns": prohibited})
        _fail(not prohibited, f"{source}: G3 held receipt contains outcome field", failures)
    # G3 is a training-only feature-selection receipt.  Some valid runs
    # deliberately persist only the strict-OOF trace and frozen contract;
    # their held score receipts live in the later fixed-role adapter.  Do not
    # turn that clean separation into a false audit failure.
    evidence["g3"] = {
        "root": root.name, "contracts": contract.get("contracts", {}),
        "receipts": rows,
        "trace_only_selection_receipt": not bool(rows),
        "run_manifest_present": manifest is not None,
        "full_window_verified": bool(manifest and manifest.get("training", {}).get("full_window_required")),
    }


def _audit_fixed_contract_query_screen(root: Path, evidence: dict[str, object], failures: list[str]) -> None:
    """Audit a LambdaRank query-localisation screen as a real query test.

    The earlier T2/T6 query receipts were L2 fits below their sampling cap, so
    different declared query names could yield identical score files.  This
    audit requires native LambdaRank group evidence *and* verifies that query
    variants actually produced a different target-free score surface.
    """
    manifest_path = root / "run_manifest.json"
    audit_path = root / "query_screen_audit.parquet"
    metric_path = root / "query_screen_metrics.parquet"
    _fail(manifest_path.exists() and audit_path.exists() and metric_path.exists(), f"{root.name}: missing query-screen receipt", failures)
    if not manifest_path.exists() or not audit_path.exists() or not metric_path.exists():
        return
    manifest = json.loads(manifest_path.read_text())
    _fail(manifest.get("schema") == "strict_r3_o3v2_fixed_contract_query_screen_v1", f"{root.name}: unexpected query-screen schema", failures)
    training = manifest.get("training", {})
    _fail(int(training.get("calendar_months", -1)) == 6, f"{root.name}: query screen training window is not six months", failures)
    _fail(int(training.get("reserve_days", -1)) == 28, f"{root.name}: query screen reserve is not 28 days", failures)
    query_modes = tuple(manifest.get("query_modes", ()))
    _fail(len(query_modes) >= 2, f"{root.name}: query screen lacks a comparison", failures)
    audit = pd.read_parquet(audit_path)
    _fail(set(audit["query_mode"].astype(str)) == set(query_modes), f"{root.name}: audit query modes do not match manifest", failures)
    _fail(audit["held_target_free"].fillna(False).astype(bool).all(), f"{root.name}: a held query receipt is not target-free", failures)
    _fail(audit["policy_labels_available_before_reserve"].fillna(False).astype(bool).all(), f"{root.name}: a query fold used post-reserve labels", failures)
    _fail(audit["query_reaches_lambdarank_loss"].fillna(False).astype(bool).all(), f"{root.name}: a query variant did not reach LambdaRank", failures)
    _fail((pd.to_numeric(audit["sampled_queries"], errors="coerce") > 0).all(), f"{root.name}: empty LambdaRank query groups", failures)
    rows: list[dict[str, object]] = []
    by_month: dict[str, dict[str, pd.DataFrame]] = {}
    for mode, source in _score_receipts(root):
        frame = pd.read_parquet(source)
        prohibited = [column for column in frame if column.startswith(PROHIBITED_SCORE_PREFIXES)]
        duplicate = int(frame["candidate_id"].duplicated().sum())
        token = source.stem.split("=", 1)[-1]
        rows.append({"mode": mode, "path": str(source), "rows": int(len(frame)), "prohibited_columns": prohibited, "duplicate_ids": duplicate})
        _fail(not prohibited, f"{source}: query receipt contains outcome fields {prohibited}", failures)
        _fail(duplicate == 0, f"{source}: duplicate query receipt IDs", failures)
        _fail("g3_rank" in frame, f"{source}: missing g3_rank", failures)
        by_month.setdefault(token, {})[mode] = frame
    _fail(bool(rows), f"{root.name}: no target-free query score receipts", failures)
    distinct_rows: list[dict[str, object]] = []
    for token, frames in sorted(by_month.items()):
        if len(frames) < 2:
            failures.append(f"{root.name} {token}: fewer than two query receipts")
            continue
        baseline_mode = query_modes[0]
        baseline = frames.get(baseline_mode)
        if baseline is None:
            failures.append(f"{root.name} {token}: missing baseline query receipt {baseline_mode}")
            continue
        base = baseline.set_index("candidate_id")["g3_rank"].astype(float)
        for mode, frame in sorted(frames.items()):
            if mode == baseline_mode:
                continue
            other = frame.set_index("candidate_id")["g3_rank"].astype(float).reindex(base.index)
            delta = np.abs(base.to_numpy(float) - other.to_numpy(float))
            changed = int(np.nansum(delta > 1e-8))
            distinct_rows.append({"month": token, "baseline": baseline_mode, "mode": mode, "changed_rows": changed, "max_abs_rank_delta": float(np.nanmax(delta))})
            _fail(changed > 0, f"{root.name} {token}: {mode} is numerically identical to {baseline_mode}", failures)
    evidence["fixed_contract_query_screen"] = {
        "root": root.name,
        "schema": manifest.get("schema"),
        "training_audit": audit.to_dict("records"),
        "target_free_receipts": rows,
        "numeric_query_differences": distinct_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantics-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--support-root", type=Path)
    parser.add_argument("--feature-root", type=Path)
    parser.add_argument("--g3-root", type=Path)
    parser.add_argument("--query-root", action="append", type=Path, default=[])
    parser.add_argument("--score-root", action="append", type=Path, default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    failures: list[str] = []
    evidence: dict[str, object] = {}
    _audit_semantics(args.semantics_root, evidence, failures)
    _audit_target(args.target_root, evidence, failures)
    if args.support_root is not None:
        _audit_support(args.support_root, evidence, failures)
    if args.feature_root is not None:
        _audit_feature_screen(args.feature_root, evidence, failures)
    if args.g3_root is not None:
        _audit_g3(args.g3_root, evidence, failures)
    for root in args.query_root:
        _audit_fixed_contract_query_screen(root, evidence, failures)
    for root in args.score_root:
        _audit_score_root(root, evidence, failures)
    report = {"schema": "strict_r3_o3v2_correctness_report_v1", "passed": not failures, "failures": failures, "evidence": evidence}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if failures:
        raise SystemExit("; ".join(failures))


if __name__ == "__main__":
    main()
