#!/usr/bin/env python3
"""Emit Stage-C correctness evidence from materialised artifacts, never assertions."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements import continuation_features as continuation

PANEL_DIR = ROOT / "data_perp/artifacts/stage_c_continuation_feature_panel_20260731_v2"
FEATURE = PANEL_DIR / "stage_c_candidate_population.parquet"
LINEAGE = PANEL_DIR / "feature_source_lineage.parquet"
GROUPS = PANEL_DIR / "retention_feature_groups.json"
MANIFEST = PANEL_DIR / "run_manifest.json"
GROUP_VALIDITY = PANEL_DIR / "feature_group_validity.parquet"
COMPATIBLE_IDS = PANEL_DIR / "stage_c_compatible_candidate_ids.parquet"
UNIVERSE = PANEL_DIR / "eligible_universe_membership.parquet"


def _id_hash(values: pd.Series) -> str:
    return hashlib.sha256("\n".join(values.astype(str).tolist()).encode("utf-8")).hexdigest()


def _result(passed: bool, evidence: str) -> dict[str, object]:
    return {"passed": bool(passed), "evidence": evidence}


def run(*, output: Path) -> dict[str, object]:
    feature = pd.read_parquet(FEATURE)
    lineage = pd.read_parquet(LINEAGE)
    groups = json.loads(GROUPS.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    group_validity = pd.read_parquet(GROUP_VALIDITY)
    compatible = pd.read_parquet(COMPATIBLE_IDS)
    universe = pd.read_parquet(UNIVERSE)
    source = inspect.getsource(continuation.materialize_ohlcv_continuation_features)
    names = [name for name in feature if name.startswith(("cont_", "side_cont_"))]
    forbidden = ("orderbook", "depth", "aggressor", "liquidation", "spread")
    blocked = lineage.loc[lineage.feature_group.isin(["F4_oi_dynamics", "F5_funding_crowding", "F7_causal_regime_transition"])]
    expected_hash = manifest["compatible_population"]["candidate_id_sha256"]
    current_hash = _id_hash(compatible.candidate_id)
    group_columns = [name for name in group_validity if name != "candidate_id"]
    compatible_validity = compatible.merge(
        group_validity,
        on="candidate_id",
        how="left",
        validate="one_to_one",
    )
    checks: dict[str, dict[str, object]] = {
        "test_all_features_available_by_decision_timestamp": _result(feature.feature_available_ts.le(feature.decision_ts).all(), "materialised feature_available_ts <= decision_ts"),
        "test_rolling_features_use_trailing_data_only": _result("center=" not in source and ".rolling(" in source, "generator source contains trailing rolling operations and no centered window"),
        "test_cross_sectional_features_use_timestamp_eligible_universe": _result({"ts", "eligible_symbol_sha256", "eligible_universe_size"}.issubset(universe.columns) and feature.cont_cs_universe_size.notna().all(), "per-timestamp eligible-universe membership digest and size are materialised"),
        "test_oi_values_respect_source_timestamp_and_staleness": _result(bool((blocked.feature_group.eq("F4_oi_dynamics") & ~blocked.point_in_time_safe.astype(bool)).any()) and groups["F4_oi_dynamics"] == [], "F4 is rejected because no source availability/staleness contract exists"),
        "test_funding_values_respect_observation_timestamp": _result(bool((blocked.feature_group.eq("F5_funding_crowding") & ~blocked.point_in_time_safe.astype(bool)).any()) and groups["F5_funding_crowding"] == [], "F5 is rejected because no source observation contract exists"),
        "test_no_future_funding_payment_used": _result(not any(any(token in name.lower() for token in ("next", "payment", "settlement")) for name in names), "no materialised feature name includes a future funding payment/settlement"),
        "test_no_inverse_pi_rows_mixed_with_linear_pf_rows": _result(feature.source_symbol.str.endswith("_USD:USD").all(), "all materialised source symbols are linear USD perpetual mappings"),
        "test_clear_first_population_matches_frozen_label_manifest": _result(feature.retain_h0_given_clear__valid.eq(feature.retain_h0_given_clear__condition_met).all(), "validity and frozen clear-first condition match"),
        "test_retention_labels_exist_only_on_clear_first_support": _result(feature.loc[~feature.retain_h0_given_clear__valid, "retain_h0_given_clear"].isna().all(), "non-clear rows have null retention labels"),
        "test_comparison_arms_use_identical_candidate_ids": _result(current_hash == expected_hash and feature.candidate_id.astype(str).equals(compatible.candidate_id.astype(str)) and compatible_validity[group_columns].all(axis=1).all(), "single complete F1/F2/F3/F6/F8 common cohort matches manifest SHA256"),
        "test_upstream_transition_predictions_are_oof": _result(groups["F7_causal_regime_transition"] == [] and bool((blocked.feature_group.eq("F7_causal_regime_transition") & ~blocked.point_in_time_safe.astype(bool)).any()), "F7 remains blocked absent verified candidate OOF/prequential sidecar"),
        "test_ohlcv_proxy_names_are_not_factual_l2": _result(all(not any(token in name.lower() for token in forbidden) or name.endswith(("_proxy", "_estimator", "_ohlcv_proxy")) for name in names), "forbidden factual-L2 tokens require an explicit proxy suffix"),
    }
    pending = {
        "test_feature_selection_uses_training_data_only": "not_run_stage1",
        "test_scalers_and_clippers_fit_on_training_data_only": "not_run_stage1",
        "test_no_final_oos_feature_selection": "not_run_stage1",
        "test_stage_b_test_changes_only_retention_head_features": "not_run_stage_b",
        "test_cost_and_execution_policy_ids_remain_frozen": "not_run_stage_b",
        "test_global_ranking_occurs_after_common_bps_mapping": "not_run_stage_b",
    }
    payload = {
        "schema": "stage_c_correctness_audit_v2",
        "stage0_passed": bool(all(item["passed"] for item in checks.values())),
        "checks": checks,
        "pending_later_stage_checks": pending,
        "inputs": {"feature_rows": len(feature), "candidate_id_sha256": current_hash, "lineage_rows": len(lineage)},
        "limitations": ["F4/F5 rejected pending native source availability timestamps", "F7 rejected pending verified OOF/prequential sidecar", "Stage1 and Stage-B checks are explicitly not run"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PANEL_DIR / "correctness_test_report.json")
    args = parser.parse_args()
    print(json.dumps(run(output=args.output), indent=2))


if __name__ == "__main__":
    main()
