#!/usr/bin/env python3
"""Fail-closed completion audit for the adaptive-exit sequential funnel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORE = ROOT / "data_perp/artifacts/adaptive_exit_sequential_funnel_20260813_v3"
DEFAULT_COMPLETION = (
    ROOT / "data_perp/artifacts/adaptive_exit_sequential_funnel_completion_20260813_v3"
)
DEFAULT_REPORT = ROOT / "docs/ADAPTIVE_EXIT_SEQUENTIAL_FUNNEL_20260813.md"

EXPECTED_CORE_ARMS = {
    "gate": {
        "G0_no_gate", "G1_raw_disagreement", "G2_action_disagreement",
        "G3_normalized_disagreement", "G4_directional_disagreement",
        "G5_action_directional", "G6_random20_placebo",
        "G7_extreme20_placebo", "G8_low_support20", "G9_high_ood20",
    },
    "direction": {
        "A0_frozen", "A1_decreases_only", "A2_increases_only", "A3_both",
        "A4_down_free_up_corroborated", "A5_up_free_down_corroborated",
        "A6_separate_direction_thresholds",
    },
    "authority": {
        "gamma_0.25", "gamma_0.5", "gamma_0.75", "gamma_1.0",
        "gamma_down50_up25", "gamma_down75_up25", "gamma_down100_up50",
        "deadband_0.00", "deadband_0.10", "deadband_0.20",
        "deadband_0.30", "deadband_0.50", "budget_10", "budget_20",
        "budget_30", "budget_50", "budget_75", "budget_100",
    },
    "uncertainty": {
        "quantile_q50", "quantile_q60", "quantile_q65", "quantile_q70",
        "quantile_q75", "uncertainty_none", "uncertainty_quantile_width",
        "uncertainty_seed_ensemble", "uncertainty_f4",
        "uncertainty_f4_plus_quantiles",
    },
    "actionable": {
        "actionable_all", "actionable_only", "schedule_first", "schedule_hour2",
        "schedule_hour4", "schedule_every_hour", "schedule_every_hour_deadband20",
    },
    "target": {
        "T1_carried_contract", "T2_increment_above_current_mfe",
        "T3_next_hour_from_current", "T4_future_peak_minus_current",
        "T5_probability_reaches_a0", "cap8", "cap10", "cap12",
        "cap_train_p99",
    },
}

EXPECTED_DIRECT = {
    "M0_linear_incumbent", "M1_isotonic", "M2_piecewise_monotonic",
    "T6_direct_huber", "T6_direct_lambdarank",
}
EXPECTED_SOURCE = {
    "S0_frozen_frozen", "S1_f1_frozen", "S2_winner_frozen",
    "S3_frozen_proxy_f1", "S4_f1_proxy_f1", "S5_winner_proxy_f1",
    "S6_winner_proxy_uncertainty",
}
FORBIDDEN_FEATURE_TOKENS = (
    "remaining_favorable", "remaining_adverse", "next_hour_favorable",
    "next_hour_adverse", "adaptive_net", "baseline_net", "oracle_",
    "raw__",
)


def _record(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any) -> None:
    checks.append({"check": name, "passed": bool(passed), "detail": detail})


def _arms(path: Path) -> set[str]:
    frame = pd.read_parquet(path)
    return set(frame["arm"].astype(str))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-dir", type=Path, default=DEFAULT_CORE)
    parser.add_argument("--completion-dir", type=Path, default=DEFAULT_COMPLETION)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    core = args.core_dir.resolve()
    completion = args.completion_dir.resolve()
    checks: list[dict[str, Any]] = []
    core_manifest = json.loads((core / "run_manifest.json").read_text())
    completion_manifest = json.loads((completion / "run_manifest.json").read_text())

    _record(checks, "completion_status", completion_manifest.get("status") == "COMPLETED", completion_manifest.get("status"))
    _record(checks, "no_automatic_promotion", completion_manifest.get("promotion") == "none", completion_manifest.get("promotion"))
    _record(checks, "development_selection_is_2025", completion_manifest["invariants"].get("selection_period") == "2025", completion_manifest["invariants"])
    _record(checks, "confirmation_is_2026", completion_manifest["invariants"].get("confirmation_period") == "2026", completion_manifest["invariants"])

    expected_policy = {
        "stop_frozen": 4.15200064332387,
        "activation_baseline": 2.326224919759605,
        "giveback_frozen": 0.10237198997143725,
        "cost_bps_once": 100,
    }
    policy_ok = all(
        abs(float(completion_manifest["invariants"][key]) - float(value)) < 1e-12
        for key, value in expected_policy.items()
    )
    _record(checks, "frozen_execution_contract", policy_ok, completion_manifest["invariants"])

    for stage, expected in EXPECTED_CORE_ARMS.items():
        selection = core / stage / "selection.parquet"
        metrics = core / stage / "metrics.parquet"
        replay = core / stage / "oof_replay.parquet"
        fit = core / stage / "fit_audit.parquet"
        present = all(path.exists() and path.stat().st_size > 0 for path in (selection, metrics, replay, fit))
        _record(checks, f"{stage}_artifacts", present, [str(path) for path in (selection, metrics, replay, fit)])
        actual = _arms(selection)
        _record(checks, f"{stage}_arms", actual == expected, {"missing": sorted(expected - actual), "extra": sorted(actual - expected)})

    simplify = _arms(core / "simplify/selection.parquet")
    simplify_required = {
        "proposal_f1_trust_positive", "proposal_f1_support_ood",
        "proposal_f1_compact_trust", "proposal_f1_trust_ev",
        *(f"veto_p{i}" for i in range(6)),
        "veto_f4_rich", "veto_f4_archetype", "veto_f4_trust",
        "veto_f4_evolution", "veto_f4_compact", "veto_f4_full",
    }
    _record(checks, "simplify_feature_and_evolution_arms", simplify == simplify_required, {"missing": sorted(simplify_required - simplify), "extra": sorted(simplify - simplify_required)})

    direct = _arms(completion / "direct_and_mapping/selection.parquet")
    source = _arms(completion / "source_factorial/source_selection.parquet")
    _record(checks, "direct_and_mapping_arms", direct == EXPECTED_DIRECT, {"missing": sorted(EXPECTED_DIRECT - direct), "extra": sorted(direct - EXPECTED_DIRECT)})
    _record(checks, "source_factorial_arms", source == EXPECTED_SOURCE, {"missing": sorted(EXPECTED_SOURCE - source), "extra": sorted(source - EXPECTED_SOURCE)})

    portability = {
        "window": {"window_3m", "window_6m", "window_9m", "window_12m", "window_expanding"},
        "weight": {"weight_uniform", "weight_recency", "weight_equal_month", "weight_equal_month_recency"},
        "regularization": {"reg_d4_l15_l2_10", "reg_d4_l15_l2_25", "reg_d4_l15_l2_50", "reg_d3_l7_l2_10", "reg_d3_l7_l2_25"},
        "missing": {"missing_native", "missing_median", "missing_median_indicators"},
        "seed": {"seed_20260811", "seed_20260812", "seed_20260813", "seed_20260814"},
    }
    for stage, expected in portability.items():
        actual = _arms(completion / "portability" / stage / "selection.parquet")
        _record(checks, f"portability_{stage}_arms", actual == expected, {"missing": sorted(expected - actual), "extra": sorted(actual - expected)})

    contracts = core_manifest["feature_contracts"]
    forbidden = {
        field for fields in contracts.values() for field in fields
        if any(token in field for token in FORBIDDEN_FEATURE_TOKENS)
    }
    _record(checks, "feature_contracts_exclude_future_labels", not forbidden, sorted(forbidden))

    final_contract = next(
        row["final_training_contract"] for row in completion_manifest["stage_results"]
        if row.get("stage") == "portability"
    )
    expected_final = {
        "window_months": 9,
        "weight_mode": "uniform",
        "proposal_feature_contract": "F1_trust_positive",
        "target": "T1_absolute_future_peak",
    }
    final_ok = all(final_contract.get(key) == value for key, value in expected_final.items())
    final_ok &= final_contract["model_spec"].get("reg_lambda") == 25
    final_ok &= final_contract["model_spec"].get("alpha") == 0.7
    _record(checks, "terminal_training_contract", final_ok, final_contract)

    final_stage = completion / "final_portability_portfolio/stage_result.json"
    final_payload = json.loads(final_stage.read_text())
    baseline = final_payload["portfolio_baseline"]
    adaptive = final_payload["portfolio_adaptive"]
    parity_ok = int(baseline["accepted_trades"]) == 8453 and abs(float(baseline["net_bps_per_trade"]) - 163.08626757132228) < 1e-8
    _record(checks, "frozen_portfolio_parity", parity_ok, baseline)
    improvement_ok = float(adaptive["net_bps_per_trade"]) > float(baseline["net_bps_per_trade"]) and int(adaptive["accepted_trades"]) >= int(baseline["accepted_trades"])
    _record(checks, "terminal_portfolio_improves_ev", improvement_ok, adaptive)

    requirement_markers = [f"| {number} |" for number in range(1, 25)]
    report_text = args.report.read_text()
    _record(checks, "report_covers_all_24_requirements", all(marker in report_text for marker in requirement_markers), str(args.report))

    passed = all(item["passed"] for item in checks)
    result = {
        "schema": "adaptive_exit_sequential_funnel_completion_audit_v1",
        "passed": passed,
        "checks_passed": sum(item["passed"] for item in checks),
        "checks_total": len(checks),
        "checks": checks,
    }
    output = completion / "correctness_test_report.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps({key: result[key] for key in ("passed", "checks_passed", "checks_total")}, indent=2))
    print(output)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
