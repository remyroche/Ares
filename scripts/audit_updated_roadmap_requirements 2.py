#!/usr/bin/env python3
"""Audit the cached target/feature/execution roadmap requirement by requirement.

This is a contract/evidence audit only.  It deliberately reads the sealed
alignment and ablation artifacts, performs no fitting, and never changes a
promotion decision.  ``PASS`` means the requirement is evidenced; ``FAIL``
means an acceptance gate is contradicted by the current data; and
``BLOCKED_EXTERNAL`` is reserved for the separate native-L2 acquisition
prerequisite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "data_perp" / "artifacts"
SCHEMA = "updated_roadmap_requirement_audit_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, pd.Timedelta, Path)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _requirement(
    requirement: str,
    section: str,
    status: str,
    evidence: str,
    observed: Any,
    rule: str,
) -> dict[str, Any]:
    return {
        "requirement": requirement,
        "section": section,
        "status": status,
        "evidence": evidence,
        "observed": _safe(observed),
        "rule": rule,
    }


def _check_map(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["check"]): row for row in report.get("checks", [])}


def _metric_row(frame: pd.DataFrame, *, arm: str, scope: str, fraction: float) -> pd.Series:
    rows = frame[(frame["arm"].astype(str) == arm) & (frame["scope"].astype(str) == scope) & np.isclose(frame["fraction"].astype(float), fraction)]
    if len(rows) != 1:
        raise ValueError(f"expected one metric row for {arm}/{scope}/{fraction}, found {len(rows)}")
    return rows.iloc[0]


def audit(
    *,
    output_dir: Path,
    alignment_dir: Path,
    target_dir: Path,
    supportive_dir: Path,
    backfill_manifest: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    correctness_path = alignment_dir / "correctness_test_report.json"
    alignment_manifest_path = alignment_dir / "run_manifest.json"
    contract_path = ARTIFACTS / "target_alignment" / "execution_target_contract.json"
    contract_view_path = alignment_dir / "candidate_target_contract.parquet"
    dictionary_path = target_dir / "label_dictionary.parquet"
    metadata_path = alignment_dir / "supportive_label_metadata.parquet"
    feature_path = alignment_dir / "feature_eligibility_manifest.parquet"
    candidate_oof_path = alignment_dir / "candidate_level_oof_predictions.parquet"
    target_metrics_path = ARTIFACTS / "exact_h12_target_purity_ablation_20260731_v11" / "target_ablation_metrics.csv"
    bootstrap_path = ARTIFACTS / "exact_h12_target_purity_ablation_20260731_v11" / "paired_day_bootstrap_vs_frozen_control.csv"
    target_results_path = ARTIFACTS / "exact_h12_target_purity_ablation_20260731_v11" / "target_ablation_results.parquet"
    policy_path = supportive_dir / "target_supportive_policy_summary.parquet"

    correctness = json.loads(correctness_path.read_text(encoding="utf-8"))
    check_map = _check_map(correctness)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract_view = pd.read_parquet(contract_view_path)
    dictionary = pd.read_parquet(dictionary_path, columns=["label_name", "model_input_allowed"])
    metadata = pd.read_parquet(metadata_path)
    features = pd.read_parquet(feature_path)
    candidate_oof = pd.read_parquet(candidate_oof_path, columns=["candidate_id", "prediction_fit_end_ts", "prediction_generated_ts", "__decision_ts__", "score"])
    target_metrics = pd.read_csv(target_metrics_path)
    target_results = pd.read_parquet(target_results_path, columns=["candidate_id", "arm", "side", "decision_ts", "exact_h12_net_bps"])
    policy = pd.read_parquet(policy_path)
    bootstrap = pd.read_csv(bootstrap_path)
    backfill = json.loads(backfill_manifest.read_text(encoding="utf-8"))

    rows: list[dict[str, Any]] = []
    required_contract = {
        "candidate_id", "symbol", "side", "decision_ts", "entry_ts", "entry_price",
        "horizon_end_ts", "label_available_ts", "row_cost_bps", "policy_geometry_id",
        "path_source", "path_complete",
    }
    rows.append(_requirement(
        "exact_target_contract_fields", "target contract", "PASS" if required_contract.issubset(contract_view.columns) else "FAIL",
        str(contract_view_path), sorted(required_contract.intersection(contract_view.columns)),
        "all roadmap identity/timing/cost/path fields are present in the canonical row view",
    ))
    rows.append(_requirement(
        "exact_h12_contract_and_cost_once", "target contract",
        "PASS" if check_map.get("exact_h12_horizon", {}).get("passed") and check_map.get("exact_net_accounting", {}).get("passed") else "FAIL",
        str(correctness_path),
        {name: check_map.get(name, {}).get("value") for name in ("exact_h12_horizon", "exact_net_accounting", "label_available_after_horizon")},
        "horizon is exactly 12 hours, labels are available at/after horizon, and net=gross-cost once",
    ))
    rows.append(_requirement(
        "shared_frozen_policy_cost_geometry", "target contract",
        "PASS" if check_map.get("single_target_policy_cost", {}).get("passed") and check_map.get("geometry_declared", {}).get("passed") else "FAIL",
        str(correctness_path), {"policy_cost": check_map.get("single_target_policy_cost", {}).get("value"), "geometry": check_map.get("geometry_declared", {}).get("value")},
        "all rows use one declared execution policy, cost model and geometry",
    ))
    rows.append(_requirement(
        "base_and_execution_targets_separated", "target layers", "PASS" if len(target_metrics["arm"].astype(str).unique()) >= 29 else "FAIL",
        str(target_metrics_path), {"arms": int(target_metrics["arm"].astype(str).nunique()), "control_present": bool(target_metrics["arm"].astype(str).eq("CONTROL_base_opportunity").any())},
        "frozen base opportunity control and the declared execution target formulations are evaluated on the same panel",
    ))
    metadata_suffixes = ("__valid", "__condition_met", "__censored", "__support_count")
    metadata_ok = all(any(str(column).endswith(suffix) for column in metadata.columns) for suffix in metadata_suffixes)
    rows.append(_requirement(
        "supportive_heads_and_metadata", "supportive labels", "PASS" if metadata_ok and check_map.get("supportive_core_heads_present", {}).get("passed") else "FAIL",
        str(metadata_path), {"rows": int(len(metadata)), "metadata_suffixes": {suffix: int(sum(str(column).endswith(suffix) for column in metadata.columns)) for suffix in metadata_suffixes}},
        "the five canonical supportive heads have explicit validity, condition, censoring and support metadata",
    ))
    rows.append(_requirement(
        "future_labels_forbidden_as_inputs", "feature eligibility", "PASS" if check_map.get("future_labels_forbidden", {}).get("passed") and check_map.get("supportive_future_labels_not_model_inputs", {}).get("passed") else "FAIL",
        str(correctness_path), {"future_labels": check_map.get("future_labels_forbidden", {}).get("value"), "dictionary_rows": int(len(dictionary)), "allowed_inputs": int(dictionary["model_input_allowed"].astype(bool).sum())},
        "future/realised labels are target-side only",
    ))
    rows.append(_requirement(
        "layer_specific_feature_eligibility", "feature eligibility", "PASS" if all(check_map.get(name, {}).get("passed") for name in ("feature_manifest_columns", "base_has_no_model_derived_inputs", "execution_derived_features_have_lineage", "action_layer_excluded")) else "FAIL",
        str(feature_path), {name: check_map.get(name, {}).get("value") for name in ("base_has_no_model_derived_inputs", "execution_derived_features_have_lineage", "action_layer_excluded")},
        "base is causal/raw-only, execution predictions have lineage, and timing/action fields are excluded",
    ))
    fit = pd.to_datetime(candidate_oof["prediction_fit_end_ts"], utc=True, errors="coerce")
    generated = pd.to_datetime(candidate_oof["prediction_generated_ts"], utc=True, errors="coerce")
    decision = pd.to_datetime(candidate_oof["__decision_ts__"], utc=True, errors="coerce")
    oof_ok = bool((fit < decision).all() and (generated <= decision).all() and np.isfinite(pd.to_numeric(candidate_oof["score"], errors="coerce")).all())
    rows.append(_requirement(
        "strict_candidate_level_oof", "OOF lineage", "PASS" if oof_ok else "FAIL",
        str(candidate_oof_path), {"rows": int(len(candidate_oof)), "fit_end_before_decision": int((fit >= decision).sum()), "generated_after_decision": int((generated > decision).sum())},
        "every upstream score is finite, fit before decision, and generated no later than decision",
    ))
    global_ok = all(check_map.get(name, {}).get("passed") for name in ("global_tail_is_pooled", "global_tail_row_count_exact", "global_tail_month_coverage"))
    rows.append(_requirement(
        "one_pooled_global_topk_policy", "candidate policy", "PASS" if global_ok else "FAIL",
        str(correctness_path), {name: check_map.get(name, {}).get("value") for name in ("global_tail_is_pooled", "global_tail_row_count_exact", "global_tail_month_coverage")},
        "selection is one pooled global top-k book, not timestamp-local or side-local tails",
    ))
    supportive_combinations = int(policy[["target_arm", "support_stage"]].drop_duplicates().shape[0])
    rows.append(_requirement(
        "controlled_target_supportive_ablation_matrix", "ablations", "PASS" if supportive_combinations >= 30 else "FAIL",
        str(policy_path), {"unique_target_support_cells": supportive_combinations, "target_arms": int(policy["target_arm"].nunique()), "support_stages": int(policy["support_stage"].nunique())},
        "primary target and cumulative supportive-label arms are evaluated with pooled economics",
    ))

    control = target_metrics[target_metrics["arm"].astype(str).eq("CONTROL_base_opportunity")]
    top10 = target_metrics[(target_metrics["scope"].astype(str) == "pooled_global_top") & np.isclose(target_metrics["fraction"].astype(float), 0.10)]
    control_top1 = _metric_row(target_metrics, arm="CONTROL_base_opportunity", scope="pooled_global_top", fraction=0.01)
    control_top5 = _metric_row(target_metrics, arm="CONTROL_base_opportunity", scope="pooled_global_top", fraction=0.05)
    latest_rows = control[(control["scope"].astype(str) == "pooled_global_membership_by_month") & control["month"].astype(str).eq("2024-11") & np.isclose(control["fraction"].astype(float), 0.10)]
    side_rows = control[(control["scope"].astype(str) == "pooled_global_membership_by_side") & np.isclose(control["fraction"].astype(float), 0.10)]
    best_supportive = float(policy["global_topk_net_bps"].astype(float).max())
    best_exact = float(top10["net_bps"].astype(float).max())
    rows.append(_requirement(
        "acceptance_gate_pooled_top10_positive", "economic acceptance", "PASS" if best_exact > 0 else "FAIL",
        str(target_metrics_path), {"best_exact_top10_net_bps": best_exact, "best_supportive_top10_net_bps": best_supportive},
        "at least one exact-H12 target arm must clear positive post-cost pooled top-10 net",
    ))
    rows.append(_requirement(
        "acceptance_gate_top1_top5_no_severe_reversal", "economic acceptance", "PASS" if min(float(control_top1["net_bps"]), float(control_top5["net_bps"])) >= -100.0 else "FAIL",
        str(target_metrics_path), {"control_top1_net_bps": float(control_top1["net_bps"]), "control_top5_net_bps": float(control_top5["net_bps"])},
        "top-1 and top-5 must not be below -100 bps under the declared gate",
    ))
    latest_net = float(latest_rows["net_bps"].iloc[0]) if len(latest_rows) == 1 else np.nan
    rows.append(_requirement(
        "acceptance_gate_latest_month_positive", "economic acceptance", "PASS" if np.isfinite(latest_net) and latest_net > 0 else "FAIL",
        str(target_metrics_path), {"latest_month": "2024-11", "control_latest_top10_net_bps": latest_net},
        "latest untouched evaluation month must be positive",
    ))
    side_nets = side_rows["net_bps"].astype(float).tolist()
    rows.append(_requirement(
        "acceptance_gate_both_sides_positive", "economic acceptance", "PASS" if len(side_nets) == 2 and min(side_nets) > 0 else "FAIL",
        str(target_metrics_path), {"control_side_top10_net_bps": side_nets},
        "neither side may exhibit a material unexplained economic failure",
    ))
    bootstrap_ok = bool(len(bootstrap) and np.isfinite(bootstrap[["delta_net_bps_p05", "delta_net_bps_p95", "probability_improves"]].to_numpy(float)).all() and (bootstrap["probability_improves"].astype(float) > 0.5).any() and (bootstrap["delta_net_bps_p05"].astype(float) > 0).any())
    rows.append(_requirement(
        "acceptance_gate_paired_day_bootstrap", "economic acceptance", "PASS" if bootstrap_ok else "FAIL",
        str(bootstrap_path), {"rows": int(len(bootstrap)), "positive_p05_arms": int((bootstrap["delta_net_bps_p05"].astype(float) > 0).sum()), "probability_improves_gt_0_5": int((bootstrap["probability_improves"].astype(float) > 0.5).sum())},
        "at least one challenger must show positive paired bootstrap lower bound and >50% improvement probability",
    ))
    rows.append(_requirement(
        "acceptance_gate_supportive_oof_incremental_value", "economic acceptance", "PASS" if best_supportive > best_exact else "FAIL",
        str(policy_path), {"best_supportive_top10_net_bps": best_supportive, "best_exact_top10_net_bps": best_exact},
        "supportive OOF predictions must improve final exact-net economics, not only label metrics",
    ))
    manifest_ok = alignment_manifest_path.exists() and all((alignment_dir / name).exists() for name in json.loads(alignment_manifest_path.read_text())["outputs_sha256"])
    rows.append(_requirement(
        "reproducible_saved_run_manifest", "auditability", "PASS" if manifest_ok else "FAIL",
        str(alignment_manifest_path), {"outputs_declared": len(json.loads(alignment_manifest_path.read_text()).get("outputs_sha256", {})), "all_declared_outputs_present": manifest_ok},
        "the sealed alignment manifest declares hashes for all published outputs",
    ))
    missing_pairs = int(backfill.get("missing_symbol_day_pairs", 0))
    rows.append(_requirement(
        "native_l2_historical_prerequisite", "external continuation gate", "PASS" if missing_pairs == 0 else "BLOCKED_EXTERNAL",
        str(backfill_manifest), {"missing_symbol_day_pairs": missing_pairs, "covered_symbol_day_pairs": backfill.get("currently_covered_symbol_day_pairs")},
        "if native-L2 continuation is used, every requested candidate symbol/day must have factual bounded-lag coverage",
    ))

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "roadmap_requirements.csv", index=False)
    counts = frame["status"].value_counts().to_dict()
    report = {
        "schema": SCHEMA,
        "status": "FAIL_CLOSED_RESEARCH_ONLY" if counts.get("FAIL", 0) or counts.get("BLOCKED_EXTERNAL", 0) else "ROADMAP_REQUIREMENTS_EVIDENCED",
        "promotion_eligible": False,
        "counts": {str(key): int(value) for key, value in counts.items()},
        "requirements": rows,
        "inputs": {str(path): sha256(path) for path in (correctness_path, alignment_manifest_path, contract_path, contract_view_path, dictionary_path, metadata_path, feature_path, candidate_oof_path, target_metrics_path, target_results_path, policy_path, bootstrap_path, backfill_manifest)},
    }
    write_json(output_dir / "report.json", report)
    sections: list[str] = ["# Updated roadmap requirement audit", "", f"- Status: **{report['status']}**", "- Promotion eligible: **false**", f"- Counts: `{json.dumps(report['counts'], sort_keys=True)}`", ""]
    for section, group in frame.groupby("section", sort=False):
        sections.extend([f"## {section}", ""])
        for item in group.itertuples(index=False):
            sections.append(f"- **{item.status}** `{item.requirement}` — {item.observed}. Evidence: `{item.evidence}`")
        sections.append("")
    sections.extend(["## Interpretation", "", "The audit separates roadmap acceptance failures from the native-L2 acquisition prerequisite. It is evidence-only and does not retrain or promote a model."])
    (output_dir / "UPDATED_ROADMAP_REQUIREMENT_AUDIT.md").write_text("\n".join(sections) + "\n", encoding="utf-8")
    outputs = {path.name: sha256(path) for path in output_dir.iterdir() if path.is_file() and path.name != "run_manifest.json"}
    manifest = {"schema": SCHEMA, "status": report["status"], "promotion_eligible": False, "report": report, "outputs_sha256": outputs}
    write_json(output_dir / "run_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS / "updated_roadmap_requirement_audit_20260801_v1")
    args = parser.parse_args()
    manifest = audit(
        output_dir=args.output_dir,
        alignment_dir=ARTIFACTS / "target_alignment" / "alignment_audit_20260801_v2",
        target_dir=ARTIFACTS / "root_cause_exact_h12_execution_target_pack_20260801_v3",
        supportive_dir=ARTIFACTS / "controlled_target_supportive_ablation_20260801_v2",
        backfill_manifest=ARTIFACTS / "native_l2_backfill_request_20260801_v1" / "run_manifest.json",
    )
    print(f"status={manifest['status']} counts={manifest['report']['counts']}")
    print(f"output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
