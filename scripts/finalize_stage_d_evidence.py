#!/usr/bin/env python3
"""Validate canonical Stage-D artifacts and emit the final sealed evidence pack."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/stage_d_final_evidence_pack_20260731_v1"
CANONICAL = {
    "oi_funding": ROOT / "data_perp/artifacts/stage_d_oi_funding_lineage_audit_20260731_v4",
    "d0": ROOT / "data_perp/artifacts/stage_d_action_counterfactuals_20260731_v2",
    "features": ROOT / "data_perp/artifacts/stage_d_action_features_20260731_v5",
    "d1": ROOT / "data_perp/artifacts/stage_d_d1_deterministic_baselines_20260731_v4",
    "d2": ROOT / "data_perp/artifacts/stage_d_action_mechanism_ablation_20260731_v4",
    "compact": ROOT / "data_perp/artifacts/stage_d_compact_action_model_20260731_v9",
    "compact_rerun": ROOT / "data_perp/artifacts/stage_d_compact_action_model_20260731_v10",
}
MANIFEST_NAMES = {
    "oi_funding": "run_manifest.json",
    "d0": "manifest.json",
    "features": "run_manifest.json",
    "d1": "manifest.json",
    "d2": "run_manifest.json",
    "compact": "run_manifest.json",
    "compact_rerun": "run_manifest.json",
}
TERMINAL = "CLEAR_EVENT_CONTINUE_EXIT_ACTION_RESEARCH_PASSES"
LINEAGE = "OI_FUNDING_CAUSAL_LINEAGE_UNRESOLVED"
NAMED_TESTS = (
    "test_action_population_is_exact_clear_first_population",
    "test_first_clear_timestamp_matches_frozen_label_pack",
    "test_action_decision_precedes_action_execution",
    "test_action_features_available_by_action_decision",
    "test_path_features_stop_at_action_decision",
    "test_future_mfe_mae_are_rejected",
    "test_exit_now_counterfactual_cost_applied_once",
    "test_continue_counterfactual_matches_frozen_policy",
    "test_delta_equals_continue_minus_exit",
    "test_action_arms_use_identical_candidate_ids",
    "test_folds_use_resolved_action_labels_only",
    "test_scalers_fit_on_training_data_only",
    "test_feature_selection_uses_training_data_only",
    "test_cross_sectional_universe_is_timestamp_eligible",
    "test_oi_requires_verified_availability_timestamp",
    "test_funding_requires_verified_availability_timestamp",
    "test_no_unbounded_oi_or_funding_forward_fill",
    "test_transition_features_require_oof_lineage",
    "test_side_outputs_are_mapped_to_incremental_bps",
    "test_action_threshold_uses_development_data_only",
    "test_no_entry_or_portfolio_policy_is_changed",
)
STAGE_D_TEST_FILES = tuple(sorted((ROOT / "tests").glob("test_*stage_d*.py")))
COMPACT_RUNNER = ROOT / "scripts/run_stage_d_compact_action_model.py"
COMPACT_TEST = ROOT / "tests/test_stage_d_compact_action_model.py"
AUTHORITATIVE_SPEC = Path("/Users/remyroche/.codex/attachments/6b4dbb8a-b97e-41a0-91f2-ca193188944a/pasted-text-1.txt")
REQUIRED_SOURCE_MANIFESTS = (
    ROOT / "data_perp/artifacts/stage_c_continuation_feature_panel_20260731_v2/run_manifest.json",
    ROOT / "data_perp/artifacts/stage_c_conditional_retention_ablation_20260731_v3/run_manifest.json",
    ROOT / "data_perp/artifacts/exact_h12_target_purity_ablation_20260731_v11/manifest.json",
    ROOT / "data_perp/artifacts/historical_exact_h12_postcost_events_20260731_v1/manifest.json",
    ROOT / "data_perp/artifacts/historical_exact_h12_postcost_persistence_labels_20260731_v1/manifest.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _resolve_declared_path(text: str) -> Path:
    path = Path(text)
    return path if path.is_absolute() else ROOT / path


def verify_source(name: str) -> dict[str, Any]:
    directory = CANONICAL[name]
    manifest_path = directory / MANIFEST_NAMES[name]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verified_inputs = 0
    for key in ("inputs", "inputs_sha256"):
        for path_text, expected in manifest.get(key, {}).items():
            path = _resolve_declared_path(path_text)
            if not path.is_file() or sha256(path) != expected:
                raise ValueError(f"{name} input hash mismatch: {path}")
            verified_inputs += 1
    verified_code = 0
    for key in ("code", "code_sha256"):
        for path_text, expected in manifest.get(key, {}).items():
            if isinstance(expected, dict):
                expected = expected.get("sha256")
            path = _resolve_declared_path(path_text)
            if not path.is_file() or sha256(path) != expected:
                raise ValueError(f"{name} code hash mismatch: {path}")
            verified_code += 1
    runner = manifest.get("runner")
    if isinstance(runner, dict):
        path = _resolve_declared_path(runner["path"])
        if not path.is_file() or sha256(path) != runner["sha256"]:
            raise ValueError(f"{name} runner hash mismatch: {path}")
        verified_code += 1
    if manifest.get("runner_sha256") is not None:
        if not COMPACT_RUNNER.is_file() or sha256(COMPACT_RUNNER) != manifest["runner_sha256"]:
            raise ValueError(f"{name} compact runner scalar hash mismatch")
        verified_code += 1
    if manifest.get("tests_sha256") is not None:
        if not COMPACT_TEST.is_file() or sha256(COMPACT_TEST) != manifest["tests_sha256"]:
            raise ValueError(f"{name} compact test scalar hash mismatch")
        verified_code += 1
    verified_outputs = 0
    for filename, expected in manifest.get("outputs_sha256", {}).items():
        path = directory / filename
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"{name} output hash mismatch: {path}")
        verified_outputs += 1
    seal_path = directory / "manifest.sha256"
    if seal_path.is_file() and seal_path.read_text(encoding="utf-8").split()[0] != sha256(manifest_path):
        raise ValueError(f"{name} manifest seal mismatch")
    return {
        "manifest": str(manifest_path.relative_to(ROOT)),
        "manifest_sha256": sha256(manifest_path),
        "schema": manifest.get("schema"),
        "status": manifest.get("status"),
        "verified_inputs": verified_inputs,
        "verified_code": verified_code,
        "verified_outputs": verified_outputs,
    }


def validate_canonical() -> dict[str, Any]:
    verified = {name: verify_source(name) for name in CANONICAL}
    compact = json.loads((CANONICAL["compact"] / "run_manifest.json").read_text())
    compact_rerun = json.loads((CANONICAL["compact_rerun"] / "run_manifest.json").read_text())
    gate = json.loads((CANONICAL["compact"] / "stage_d_action_research_gate.json").read_text())
    lineage = json.loads((CANONICAL["oi_funding"] / "run_manifest.json").read_text())
    groups = json.loads((CANONICAL["features"] / "stage_d_action_feature_groups.json").read_text())
    population = json.loads((CANONICAL["d0"] / "stage_d_action_population_manifest.json").read_text())
    if compact.get("terminal_decision") != TERMINAL or gate.get("terminal_decision") != TERMINAL:
        raise ValueError("compact v9 terminal decision is missing or inconsistent")
    if compact.get("outputs_sha256") != compact_rerun.get("outputs_sha256"):
        raise ValueError("compact v9/v10 same-code rerun outputs are not byte-identical")
    if compact.get("runner_sha256") != compact_rerun.get("runner_sha256"):
        raise ValueError("compact v9/v10 runner hashes differ")
    if not all(gate.get("gates", {}).values()):
        raise ValueError("compact v9 research gate is incomplete")
    if compact.get("status") != "RESEARCH_ONLY_NO_ENTRY_OR_PORTFOLIO_POLICY_CHANGE":
        raise ValueError("compact v9 changes or fails to freeze entry/portfolio policy")
    if lineage.get("a6_disposition") != "REJECTED_LINEAGE" or lineage.get("a7_disposition") != "REJECTED_LINEAGE":
        raise ValueError("OI/funding lineage was improperly admitted")
    if lineage.get("companion_disposition") != LINEAGE:
        raise ValueError("lineage companion disposition mismatch")
    if groups.get("common_population", {}).get("rows") != 108139 or population.get("eligible_action_rows") != 108139:
        raise ValueError("canonical population row mismatch")
    if compact.get("source_population_rows") != 108139:
        raise ValueError("compact population mismatch")
    stage_b = (ROOT / "ROADMAP_ABLATION_STATUS_20260731.md").read_text(encoding="utf-8")
    stage_c = (ROOT / "STAGE_C_FINAL_REPORT_20260731.md").read_text(encoding="utf-8")
    if "STAGE_B_NO_EXECUTION_TARGET_ADVANCES" not in stage_b:
        raise ValueError("frozen Stage-B decision missing")
    if "CURRENT_OHLCV_OI_FUNDING_CONTRACT_INSUFFICIENT_FOR_ENTRY_RETENTION" not in stage_c:
        raise ValueError("frozen Stage-C decision missing")
    return {
        "canonical_artifacts": verified,
        "population_rows": 108139,
        "terminal_decision": TERMINAL,
        "lineage_disposition": LINEAGE,
        "stage_b_decision": "STAGE_B_NO_EXECUTION_TARGET_ADVANCES",
        "stage_c_decision": "CURRENT_OHLCV_OI_FUNDING_CONTRACT_INSUFFICIENT_FOR_ENTRY_RETENTION",
        "no_entry_or_portfolio_policy_change": True,
    }


def discover_and_run_named_tests() -> dict[str, Any]:
    node_ids: list[str] = []
    found_names: set[str] = set()
    pattern = re.compile(r"^def (test_[A-Za-z0-9_]+)\(", re.MULTILINE)
    for path in STAGE_D_TEST_FILES:
        for name in pattern.findall(path.read_text(encoding="utf-8")):
            if name in NAMED_TESTS:
                node_ids.append(f"{path.relative_to(ROOT)}::{name}")
                found_names.add(name)
    missing = sorted(set(NAMED_TESTS) - found_names)
    if missing:
        raise ValueError(f"required named tests missing: {missing}")
    command = [sys.executable, "-m", "pytest", "-q", *node_ids]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise ValueError(f"exact named tests failed rc={completed.returncode}\n{output}")
    return {
        "return_code": completed.returncode,
        "unique_required_names": len(NAMED_TESTS),
        "executed_node_count": len(node_ids),
        "node_ids": node_ids,
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in STAGE_D_TEST_FILES},
        "command": command,
        "output": output,
    }


def validate_report_metrics() -> dict[str, Any]:
    d1 = pd.read_parquet(CANONICAL["d1"] / "stage_d_d1_baseline_summary.parquet")
    baseline = d1.loc[(d1.group_type == "overall") & (d1.group_value == "ALL")].iloc[0]
    compact = pd.read_parquet(CANONICAL["compact"] / "stage_d_compact_model_results.parquet")
    chosen = compact.loc[compact.arm == "compact_readmitted"]
    dev = chosen.loc[(chosen.split == "development_oof") & (chosen.margin_bps == 0) & (chosen.dimension == "overall")].iloc[0]
    final = chosen.loc[(chosen.split == "final_oos") & (chosen.margin_bps == 0) & (chosen.dimension == "overall")].iloc[0]
    sides = chosen.loc[(chosen.split == "final_oos") & (chosen.margin_bps == 0) & (chosen.dimension == "side")].set_index("value")
    latest = chosen.loc[(chosen.split == "final_oos") & (chosen.margin_bps == 0) & (chosen.dimension == "latest_period")].iloc[0]
    bootstrap = json.loads((CANONICAL["compact"] / "stage_d_action_replay_bootstrap.json").read_text())
    gate = json.loads((CANONICAL["compact"] / "stage_d_action_research_gate.json").read_text())
    feature_manifest = json.loads((CANONICAL["compact"] / "stage_d_compact_feature_manifest.json").read_text())
    d1_contract = json.loads((CANONICAL["d1"] / "stage_d_d1_evaluation_contract.json").read_text())
    d2_selection = json.loads((CANONICAL["d2"] / "stage_d_d9_development_selection.json").read_text())
    d2_results = pd.read_parquet(CANONICAL["d2"] / "stage_d_action_model_results.parquet")
    d2_aggregate = d2_results.loc[(d2_results.split == "development_oof") & (d2_results.dimension == "aggregate")].set_index("arm")
    oi_tests = json.loads((CANONICAL["oi_funding"] / "oi_funding_availability_tests.json").read_text())
    day_bootstrap = d1_contract["bootstrap"]
    assertions = {
        "population_rows": int(baseline.rows) == 108139,
        "always_continue_net": abs(float(baseline.continue_net_mean_bps) - 27.745751) < 1e-6,
        "always_exit_net": abs(float(baseline.exit_net_mean_bps) - 7.093963) < 1e-6,
        "exit_minus_continue": abs(float(baseline.exit_minus_continue_mean_bps) + 20.651788) < 1e-6,
        "d1_loss_accounting": abs(float(baseline.loss_avoided_sum_bps) - 8306201.079522) < 1e-6 and abs(float(baseline.false_exit_opportunity_cost_sum_bps) - 10539464.822358) < 1e-6,
        "d1_distribution": abs(float(baseline.exit_better_row_rate) - .407642) < 1e-6 and abs(float(baseline.loss_avoided_mean_bps) - 76.810411) < 1e-6 and abs(float(baseline.loss_avoided_p90_bps) - 258.791033) < 1e-6 and abs(float(baseline.false_exit_opportunity_cost_mean_bps) - 97.462200) < 1e-6 and abs(float(baseline.false_exit_opportunity_cost_p90_bps) - 277.806715) < 1e-6,
        "d1_bootstrap": day_bootstrap["bootstrap_reps"] == 2000 and day_bootstrap["seed"] == 20260731 and day_bootstrap["utc_day_blocks"] == 611,
        "d1_bootstrap_interval": all(abs(a - b) < 1e-6 for a, b in zip(day_bootstrap["paired_utc_day_block_bootstrap_95pct_ci_bps"], [-25.223851, -16.468539])),
        "development_rows": int(dev.rows) == 24267,
        "final_rows": int(final.rows) == 31258,
        "final_net": abs(float(final.net_policy_bps) - 104.849168) < 1e-6,
        "final_uplifts": abs(float(final.incremental_vs_continue_bps) - 80.122798) < 1e-6 and abs(float(final.incremental_vs_exit_bps) - 98.616308) < 1e-6,
        "development_economics": abs(float(dev.net_policy_bps) - 94.310198) < 1e-6 and abs(float(dev.incremental_vs_continue_bps) - 75.563426) < 1e-6 and abs(float(dev.incremental_vs_exit_bps) - 98.525367) < 1e-6,
        "final_diagnostics": abs(float(final.mae_bps) - 130.206289) < 1e-6 and abs(float(final.spearman_ic) - .839426) < 1e-6 and abs(float(final.roc_auc) - .961200) < 1e-6 and abs(float(final.pr_auc) - .978004) < 1e-6 and abs(float(final.brier) - .066812) < 1e-6 and abs(float(final.log_loss) - .227746) < 1e-6 and abs(float(final.calibration_slope) - 1.118887) < 1e-6 and abs(float(final.calibration_intercept_bps) + 9.831173) < 1e-6,
        "final_action_and_error_costs": abs(float(final.continue_rate) - .540182) < 1e-6 and abs(float(final.giveback_cases_exited_pct) - .976485) < 1e-6 and abs(float(final.retained_cases_incorrectly_exited_pct) - .140046) < 1e-6 and abs(float(final.loss_avoided_bps) - 81.658692) < 1e-6 and abs(float(final.false_exit_opportunity_cost_bps) - 1.535894) < 1e-6,
        "side_uplifts": abs(float(sides.loc["long", "incremental_vs_continue_bps"]) - 90.448584) < 1e-6 and abs(float(sides.loc["long", "incremental_vs_exit_bps"]) - 109.223317) < 1e-6 and abs(float(sides.loc["short", "incremental_vs_continue_bps"]) - 68.487407) < 1e-6 and abs(float(sides.loc["short", "incremental_vs_exit_bps"]) - 86.664028) < 1e-6,
        "latest_period": str(latest.value) == "2024-11" and abs(float(latest.incremental_vs_continue_bps) - 88.971227) < 1e-6 and abs(float(latest.incremental_vs_exit_bps) - 121.843392) < 1e-6,
        "every_final_month_positive": bool(((chosen.split == "final_oos") & (chosen.margin_bps == 0) & (chosen.dimension == "month")).any()) and bool((chosen.loc[(chosen.split == "final_oos") & (chosen.margin_bps == 0) & (chosen.dimension == "month"), ["incremental_vs_continue_bps", "incremental_vs_exit_bps"]] > 0).all().all()),
        "bootstrap": bootstrap["reps"] == 1000 and bootstrap["seed"] == 20260731,
        "bootstrap_positive": bootstrap["versus_always_continue"]["prob_positive"] == 1.0 and bootstrap["versus_always_exit"]["prob_positive"] == 1.0,
        "bootstrap_intervals": all(abs(a - b) < 1e-6 for a, b in zip(bootstrap["versus_always_continue"]["ci_95_bps"], [75.845017, 84.670889])) and all(abs(a - b) < 1e-6 for a, b in zip(bootstrap["versus_always_exit"]["ci_95_bps"], [90.663847, 106.279223])),
        "all_gates": all(gate["gates"].values()),
        "symbol_stability": gate["symbol_stability"]["symbol_support"] == 126 and abs(gate["symbol_stability"]["max_absolute_symbol_uplift_concentration"] - .015832451) < 1e-9,
        "a0_only": feature_manifest["included_groups_after_readmission"] == ["A0_minimal_action_state_control"],
        "terminal": gate["terminal_decision"] == TERMINAL,
        "oi_funding_blocked": oi_tests["source_classes"] == 13 and oi_tests["admitted_sources"] == 0,
        "d2_a1_only": d2_selection["approved"] == ["A1"],
        "d2_report_diagnostics": (
            abs(float(d2_aggregate.loc["D0", "mae_bps"]) - 178.555338) < 1e-6
            and abs(float(d2_aggregate.loc["D1", "mae_bps"]) - 173.866769) < 1e-6
            and abs(float(d2_aggregate.loc["D0", "spearman_ic"]) - 0.614814) < 1e-6
            and abs(float(d2_aggregate.loc["D1", "spearman_ic"]) - 0.639548) < 1e-6
            and abs(float(d2_aggregate.loc["D0", "roc_auc"]) - 0.847050) < 1e-6
            and abs(float(d2_aggregate.loc["D1", "roc_auc"]) - 0.867895) < 1e-6
            and abs(float(d2_aggregate.loc["D0", "brier"]) - 0.154026) < 1e-6
            and abs(float(d2_aggregate.loc["D1", "brier"]) - 0.151319) < 1e-6
        ),
    }
    if not all(assertions.values()):
        raise ValueError(f"report metric assertion failure: {assertions}")
    return {
        "assertions": assertions,
        "baseline": baseline.to_dict(),
        "development_overall": dev.to_dict(),
        "final_overall": final.to_dict(),
        "final_sides": {str(index): row.to_dict() for index, row in sides.iterrows()},
        "latest_period": latest.to_dict(),
        "bootstrap": bootstrap,
        "d1_bootstrap": day_bootstrap,
        "d2_selection": d2_selection,
        "d2_development_aggregate": {str(index): row.to_dict() for index, row in d2_aggregate.iterrows()},
        "oi_funding": oi_tests,
        "gate": gate,
        "development_period": "2024-04..2024-07",
        "final_oos_period": "2024-08..2024-11",
        "folds": 4,
        "model_seed": 20260731,
    }


def _disposition_yaml() -> str:
    return """schema: stage_d_feature_disposition_v1
population_rows: 108139
model_terminal_decision: CLEAR_EVENT_CONTINUE_EXIT_ACTION_RESEARCH_PASSES
optional_lineage_disposition: OI_FUNDING_CAUSAL_LINEAGE_UNRESOLVED
entry_or_portfolio_policy_changed: false
groups:
  A0:
    name: minimal_action_state_control
    disposition: CONTROL
    evidence: fixed action-layer baseline
  A1:
    name: path_geometry_to_clear
    disposition: DROPPED_COMPACT_READMISSION
    evidence: passed cumulative D2 development screen, but A0-only outperformed A0+A1 on identical development rows under compact readmission
  A2:
    name: candle_rejection_structure
    disposition: REJECTED_DEVELOPMENT_GATE
    evidence: prediction did not improve
  A3:
    name: volume_confirmation_to_clear
    disposition: REJECTED_SOURCE_UNAVAILABLE
    evidence: exact one-minute paths contain no volume and no immutable aligned replacement was proven
  A4:
    name: volatility_instability_to_clear
    disposition: REJECTED_DEVELOPMENT_GATE
    evidence: incremental policy effect was negative
  A5:
    name: market_cross_sectional_confirmation
    disposition: REJECTED_DEVELOPMENT_GATE
    evidence: prediction and policy increment failed
  A6:
    name: open_interest_path
    disposition: REJECTED_LINEAGE
    evidence: observation/availability timestamps and bounded staleness remain unproven
  A7:
    name: funding_path_crowding
    disposition: REJECTED_LINEAGE
    evidence: availability and settlement semantics remain unproven
  A8:
    name: regime_transition_context
    disposition: REJECTED_OOF_LINEAGE
    evidence: no candidate/action-level strict OOF or prequential sidecar was admitted
  A9:
    name: compact_composites
    disposition: REJECTED_DEVELOPMENT_GATE
    evidence: short-only effect; long-side increment was negative
"""


def _report() -> str:
    return """# Stage-D final report — 2026-07-31

## Verdict and scope

`CLEAR_EVENT_CONTINUE_EXIT_ACTION_RESEARCH_PASSES`

Optional data-lineage disposition: `OI_FUNDING_CAUSAL_LINEAGE_UNRESOLVED`.

This is a research-only pass for the binary action taken after an exact clear event: `EXIT_NOW` versus `CONTINUE_FROZEN_POLICY`. It does not change or validate candidate entry, the frozen Stage-B hierarchy, entry thresholds, sizing, exposure, concurrency, or portfolio policy. The frozen decisions `STAGE_B_NO_EXECUTION_TARGET_ADVANCES` and `CURRENT_OHLCV_OI_FUNDING_CONTRACT_INSUFFICIENT_FOR_ENTRY_RETENTION` remain in force. Final-OOS action results are an assessment of the development-frozen rule, not a new selection surface and not evidence that the underlying entry system is deployable.

Canonical evidence is D0-v2, corrected action features v5, deterministic baselines v4, mechanism ablation v4, remediated compact action model v9, and OI/funding lineage v4. Compact v10 is the byte-identical same-code rerun of v9. Their declared input, code, output, and external manifest seals verify against current bytes.

## Answers to the thirteen required questions

1. **Is always exiting at first clear better than always continuing?** No. Across 108,139 actionable clear-first rows, always continue averages 27.746 net bps/trade and always exit 7.094 bps/trade. Signed `EXIT_NOW − CONTINUE` is −20.652 bps/trade; the paired 2,000-replicate UTC-day interval is [−25.224, −16.469].

2. **How large is giveback under the frozen policy?** Exiting is better on 40.764% of rows. The positive loss that mechanical exits could avoid sums to 8,306,201 bps, or 76.810 unconditional bps/trade (p90 258.791). This is not the signed policy effect: mechanical exit also sacrifices 10,539,465 bps of retained upside, 97.462 unconditional bps/trade (p90 277.807), which is why always exit loses overall.

3. **Does observed path-to-clear improve continuation prediction?** The broader A1 path group improves the cumulative D2 development diagnostics (MAE 178.555→173.867 bps, Spearman IC .615→.640, ROC-AUC .847→.868, Brier .1540→.1513), but it does **not** survive the stricter compact readmission on identical development rows. The final compact model is A0-only; no unsupported A1 claim is carried forward.

4. **Which mechanisms add information?** A1 passed the cumulative D2 screen but was dropped at compact readmission because A0-only performed better. Therefore no add-on group enters the final compact model. A2 failed prediction improvement; A4 had a negative policy increment; A5 failed prediction and policy value; A9 had a negative long-side effect. A3 was unavailable. A6/A7/A8 were correctly not run.

5. **Are improvements stable by side, month, symbol, and time-to-clear?** The A0-only compact final assessment is positive versus both baselines on long (+90.449/+109.223 bps) and short (+68.487/+86.664 bps), covers 126 symbols, and has maximum absolute symbol-uplift concentration .0158. Every final month is positive versus both baselines. Time-to-clear and symbol slices are reported without reranking and do not alter the fixed rule.

6. **Were OI and funding admitted?** No. A6 and A7 are `REJECTED_LINEAGE`; 13 source classes were audited with zero admissions. Missing availability timestamps, unbounded fill, product ambiguity, mixed units, and absent funding settlement semantics prevent causal use.

7. **Does the learned action policy improve net versus both deterministic baselines?** Yes for the frozen 0-bps margin. Development net is 94.310 bps/trade, +75.563 versus continue and +98.525 versus exit. Final-OOS net is 104.849 bps/trade, +80.123 versus continue and +98.616 versus exit. This is conditional action-layer economics on already-clear rows.

8. **How much loss is avoided by correct exits?** Final OOS avoids 81.659 unconditional bps/trade through correct exits and exits 97.649% of giveback cases.

9. **How much retained upside is sacrificed by false exits?** Final OOS false-exit opportunity cost is 1.536 unconditional bps/trade; 14.005% of retained cases are incorrectly exited.

10. **Does the latest period pass?** Yes. The development-selected 0-bps rule yields November 2024 uplift of +88.971 bps versus always continue and +121.843 bps versus always exit. No final month was used to choose the rule or margin.

11. **What is paired day-block uncertainty?** With 1,000 fixed UTC-day bootstrap replicates, final policy uplift has 95% intervals [75.845, 84.671] bps versus always continue and [90.664, 106.279] bps versus always exit; both positive probabilities are 1.0.

12. **What is the terminal decision?** Exactly one model decision: `CLEAR_EVENT_CONTINUE_EXIT_ACTION_RESEARCH_PASSES`. `OI_FUNDING_CAUSAL_LINEAGE_UNRESOLVED` is separately recorded as an optional data-lineage disposition, not a second model decision.

13. **What remains blocked because no entry model passed?** Entry-target promotion, a deployable entry score, candidate entry thresholds, Stage-B substitution, global entry ranking, sizing, stops, exposure, concurrency, and portfolio promotion all remain blocked. A Stage-D action pass cannot establish profitability or deployability of the upstream candidate population.

## Compact model and validation

The compact model contains A0 only after development-only re-admission dropped A1, with training-only clipping, correlation reduction and side-local feature selection capped at 32. The action margin is 0 bps, selected on 24,267 development-OOF rows and frozen before 31,258 final-OOS rows. Final diagnostics are MAE 130.206 bps, Spearman IC .839, ROC-AUC .961, PR-AUC .978, Brier .0668, log loss .2277, calibration slope 1.119 and intercept −9.831 bps. The continue/exit rates are 54.018%/45.982%. Final and development leave-group-out/symbol slices use identical candidate hashes and rows.

All 21 specification-named correctness tests pass exactly. They cover population identity, first-clear timing, causal execution, feature cutoffs, future-path exclusion, one-time costs, frozen continuation, exact delta arithmetic, arm identity, resolved-label folds, training-only preprocessing/selection, eligible cross sections, OI/funding rejection and bounded age, transition OOF lineage, incremental-bps mapping, development-only thresholding, and unchanged entry/portfolio policy.

## Protocol, gates, blocked stages, and limitations

Development-only selection uses four monthly OOF folds from 2024-04 through 2024-07; the untouched descriptive final-OOS period is 2024-08 through 2024-11. The model seed and both paired bootstrap seeds are 20260731; D1 uses 2,000 UTC-day replicates across 611 days and the final compact replay uses 1,000 replicates. Counts are 108,139 source clear-first rows, 24,267 development rows, 31,258 final rows, and 126 final symbols.

All eight frozen research gates pass: causal/lineage integrity, both paired baseline uplifts, side stability, latest-period uplift, calibration, action support, and symbol breadth/concentration. D3/A3, D6/A6, D7/A7, and D8/A8 remain blocked by source or lineage constraints. The evidence is candidate-conditioned, research-only, and final-OOS descriptive; it does not validate entry quality, remove upstream selection bias, authorize policy changes, or establish live trading performance.
"""


def run(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"fresh output root required: {output}")
    evidence = validate_canonical()
    named_test_run = discover_and_run_named_tests()
    metrics = validate_report_metrics()
    output.mkdir(parents=True)
    report = _report()
    dispositions = _disposition_yaml()
    tests = {
        "schema": "stage_d_final_correctness_report_v1",
        "passed": True,
        "exact_named_test_count": len(NAMED_TESTS),
        "exact_named_tests": {name: True for name in NAMED_TESTS},
        "exact_test_execution": named_test_run,
        "report_metric_validation": metrics,
        "artifact_hash_verification": evidence["canonical_artifacts"],
        "population_rows": evidence["population_rows"],
        "terminal_decision_count": 1,
        "terminal_decision": TERMINAL,
        "optional_lineage_disposition": LINEAGE,
        "frozen_stage_b_decision": evidence["stage_b_decision"],
        "frozen_stage_c_decision": evidence["stage_c_decision"],
        "no_entry_or_portfolio_policy_change": True,
    }
    _atomic_text(output / "STAGE_D_FINAL_REPORT.md", report)
    _atomic_text(output / "stage_d_feature_disposition.yaml", dispositions)
    _atomic_text(output / "correctness_test_report.json", json.dumps(tests, indent=2, sort_keys=True) + "\n")
    _atomic_text(ROOT / "STAGE_D_FINAL_REPORT.md", report)
    inputs = {
        value["manifest"]: value["manifest_sha256"]
        for value in evidence["canonical_artifacts"].values()
    }
    inputs.update({
        str(AUTHORITATIVE_SPEC): sha256(AUTHORITATIVE_SPEC),
        "ROADMAP_ABLATION_STATUS_20260731.md": sha256(ROOT / "ROADMAP_ABLATION_STATUS_20260731.md"),
        "STAGE_C_FINAL_REPORT_20260731.md": sha256(ROOT / "STAGE_C_FINAL_REPORT_20260731.md"),
        "STAGE_C_COMPLETION_TODO_20260731.md": sha256(ROOT / "STAGE_C_COMPLETION_TODO_20260731.md"),
        "TARGET_AUDIT_20260731.md": sha256(ROOT / "TARGET_AUDIT_20260731.md"),
        "TARGET_AUDIT_20260731_EXACT_PERSISTENCE.md": sha256(ROOT / "TARGET_AUDIT_20260731_EXACT_PERSISTENCE.md"),
    })
    inputs.update({str(path.relative_to(ROOT)): sha256(path) for path in REQUIRED_SOURCE_MANIFESTS})
    relevant_code = {Path(__file__).resolve(), COMPACT_RUNNER, *STAGE_D_TEST_FILES}
    for name in CANONICAL:
        manifest = json.loads((CANONICAL[name] / MANIFEST_NAMES[name]).read_text())
        if isinstance(manifest.get("runner"), dict):
            relevant_code.add(_resolve_declared_path(manifest["runner"]["path"]))
        for path_text in manifest.get("code", {}):
            relevant_code.add(_resolve_declared_path(path_text))
        for path_text in manifest.get("code_sha256", {}):
            relevant_code.add(_resolve_declared_path(path_text))
    outputs = {
        path.name: sha256(path)
        for path in sorted(output.iterdir())
        if path.is_file()
    }
    manifest = {
        "schema": "stage_d_final_evidence_pack_v1",
        "status": "SEALED_RESEARCH_ONLY_NO_ENTRY_OR_PORTFOLIO_POLICY_CHANGE",
        "terminal_model_decision": TERMINAL,
        "terminal_model_decision_count": 1,
        "optional_lineage_disposition": LINEAGE,
        "population_rows": 108139,
        "exact_named_tests_passed": 21,
        "periods": {"development_oof": "2024-04..2024-07", "final_oos_descriptive_only": "2024-08..2024-11"},
        "folds": {"development_monthly_oof": 4, "final_oos": 1},
        "seeds": {"model": 20260731, "d1_utc_day_bootstrap": 20260731, "compact_utc_day_bootstrap": 20260731},
        "counts": {"source_population_rows": 108139, "development_rows": 24267, "final_rows": 31258, "final_symbol_support": 126, "d1_bootstrap_reps": 2000, "compact_bootstrap_reps": 1000},
        "gates": metrics["gate"]["gates"],
        "blocked_stages": {"D3_A3": "REJECTED_SOURCE_UNAVAILABLE", "D6_A6": "REJECTED_LINEAGE", "D7_A7": "REJECTED_LINEAGE", "D8_A8": "REJECTED_OOF_LINEAGE"},
        "limitations": ["candidate_conditioned_action_layer_only", "research_only", "final_oos_descriptive_only", "no_entry_quality_validation", "no_live_trading_claim", "no_entry_or_portfolio_policy_change"],
        "inputs_sha256": dict(sorted(inputs.items())),
        "code_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sorted(relevant_code)},
        "outputs_sha256": outputs,
        "root_report_sha256": sha256(ROOT / "STAGE_D_FINAL_REPORT.md"),
        "manifest_self_hash_excluded": True,
    }
    _atomic_text(output / "run_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    for filename, expected in manifest["outputs_sha256"].items():
        if sha256(output / filename) != expected:
            raise ValueError(f"final output hash mismatch: {filename}")
    _atomic_text(output / "manifest.sha256", f"{sha256(output / 'run_manifest.json')}  run_manifest.json\n")
    return {
        "output": str(output),
        "source_manifests": len(inputs),
        "outputs_hashed": len(outputs),
        "named_tests_passed": 21,
        "terminal_decision": TERMINAL,
        "lineage_disposition": LINEAGE,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
