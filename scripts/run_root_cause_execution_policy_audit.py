#!/usr/bin/env python3
"""Stage 5/6 root-cause execution and policy diagnostics.

This runner is deliberately diagnostic-only.  It never manufactures an ideal
entry, delayed fill, spread, or slippage series when the frozen substrate does
not contain one, and it never trains an action head.  Missing counterfactuals
are emitted as explicit NOT_RUN rows so the waterfall cannot silently turn an
unidentified component into zero loss.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = ROOT / "data_perp/artifacts/root_cause_diagnostic_substrate_20260731_v4/diagnostic_row_ledger.parquet"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/root_cause_execution_policy_audit_20260731_v4"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _id_sha(frame: pd.DataFrame) -> str:
    payload = "\n".join(frame.candidate_id.astype(str)).encode()
    return hashlib.sha256(payload).hexdigest()


def _top_global(frame: pd.DataFrame, score: str, fraction: float = 0.10) -> pd.DataFrame:
    """Select global top-k, never per timestamp."""
    usable = frame.loc[frame[score].notna()].copy()
    count = max(1, int(np.ceil(len(usable) * fraction)))
    # Exact top-k support is part of the evaluation contract.  A deterministic
    # candidate-ID tie break prevents a large tied score band from silently
    # inflating the selected population.
    usable["__candidate_tie__"] = usable.candidate_id.astype(str)
    selected = usable.sort_values(
        [score, "__candidate_tie__"], ascending=[False, True], kind="stable"
    ).head(count)
    return selected.drop(columns="__candidate_tie__")


def execution_waterfall(ledger: pd.DataFrame) -> pd.DataFrame:
    gross = ledger.execution_adjusted_gross_h12_bps
    fee = ledger.fee_bps
    net = ledger.net_h12_bps
    rows: list[dict] = []

    def add(stage: str, status: str, value: float | None, detail: str) -> None:
        rows.append({
            "record_type": "waterfall",
            "stage": stage,
            "slice": "full_population",
            "score": "none",
            "rows": len(ledger),
            "value_bps_per_candidate": value,
            "status": status,
            "detail": detail,
        })

    add("A_reference_ideal_entry_gross", "NOT_RUN_SOURCE_UNAVAILABLE", None,
        "No immutable pre-spread ideal-entry outcome exists on the frozen rows.")
    add("B_executable_entry_gross", "OBSERVED", float(gross.mean()),
        "Execution-adjusted, pre-fee H12 outcome; spread/slippage are embedded in fills.")
    add("C_delayed_entry_gross", "NOT_RUN_SOURCE_UNAVAILABLE", None,
        "No immutable 1/5/10-minute delayed-entry tail exists for these rows.")
    add("D_frozen_policy_gross", "OBSERVED_NOT_SEPARABLE_FROM_B", float(gross.mean()),
        "Observed outcome already uses frozen exit geometry; entry and geometry effects cannot be separated.")
    add("E_post_cost_net", "OBSERVED_FUTURE_RESOLVED_FEE", float(net.mean()),
        "Diagnostic realised net only; row fee is unavailable at decision time.")
    add("cost_drag_D_minus_E", "IDENTIFIED", float(fee.mean()),
        "Mean realised fee; reconciliation is D - fee = E.")
    add("entry_transfer_loss_A_minus_B", "NOT_IDENTIFIABLE", None, "A is unavailable.")
    add("delay_slippage_loss_B_minus_C", "NOT_IDENTIFIABLE", None, "C is unavailable.")
    add("policy_geometry_loss_C_minus_D", "NOT_IDENTIFIABLE", None, "C is unavailable and B/D are bundled.")

    # Candidate-selection results use global top-k only.  These are observed
    # executable outcomes, not substitutes for the missing reference oracle.
    for score in ("score_base_alpha", "score_residual_alpha"):
        selected = _top_global(ledger, score)
        rows.append({
            "record_type": "global_top10_observed",
            "stage": "B_executable_entry_gross",
            "slice": "global_top10",
            "score": score,
            "rows": len(selected),
            "value_bps_per_candidate": float(selected.execution_adjusted_gross_h12_bps.mean()),
            "status": "OBSERVED",
            "detail": "Global top 10% across the full evaluation population; not per timestamp.",
        })
        rows.append({
            "record_type": "global_top10_observed",
            "stage": "E_post_cost_net",
            "slice": "global_top10",
            "score": score,
            "rows": len(selected),
            "value_bps_per_candidate": float(selected.net_h12_bps.mean()),
            "status": "OBSERVED_FUTURE_RESOLVED_FEE",
            "detail": "Realised diagnostic net on the identical global top-10 rows.",
        })

    # Controlled cost-only factorial.  Unavailable dimensions stay explicit.
    for model, cost in (
        ("zero_fee", pd.Series(0.0, index=ledger.index)),
        ("fixed_ex_ante_fee", ledger.fixed_ex_ante_fee_bps),
        ("realised_fee_diagnostic_only", fee),
    ):
        rows.append({
            "record_type": "factorial",
            "stage": "cost_model",
            "slice": model,
            "score": "none",
            "rows": len(ledger),
            "value_bps_per_candidate": float((gross - cost).mean()),
            "status": "RUN" if model != "realised_fee_diagnostic_only" else "RUN_DIAGNOSTIC_NOT_CAUSAL",
            "detail": "Entry convention/path resolution/delay/exit geometry held at the only observed frozen setting.",
        })
    for dimension in ("entry_convention", "path_resolution", "delay", "frozen_exit_geometry"):
        rows.append({
            "record_type": "factorial",
            "stage": dimension,
            "slice": "alternative",
            "score": "none",
            "rows": 0,
            "value_bps_per_candidate": None,
            "status": "NOT_RUN_SOURCE_UNAVAILABLE",
            "detail": "No identical-row immutable alternative on the canonical substrate.",
        })
    return pd.DataFrame(rows)


def policy_regret(ledger: pd.DataFrame) -> pd.DataFrame:
    # The canonical materializer names the clean gross-arm availability status
    # explicitly; accept the stable AVAILABLE_ prefix so schema wording does
    # not erase valid counterfactual rows.
    eligible = ledger.loc[ledger.action_target_status.astype(str).str.startswith("AVAILABLE_")].copy()
    rows: list[dict] = []

    def add(population: str, policy: str, count: int, gross: float | None,
            oracle_regret: float | None, status: str, detail: str) -> None:
        rows.append({
            "population": population,
            "policy": policy,
            "rows": count,
            "gross_bps_per_candidate": gross,
            "oracle_regret_bps_per_candidate": oracle_regret,
            "status": status,
            "detail": detail,
        })

    if len(eligible):
        cont = eligible.action_continue_execution_adjusted_gross_bps
        alt = eligible.action_exit_execution_adjusted_gross_bps
        oracle = np.maximum(cont, alt)
        add("conditional_eligible", "always_baseline_continue", len(eligible), float(cont.mean()),
            float((oracle - cont).mean()), "OBSERVED", "Clean gross counterfactual; no realised row cost in target.")
        add("conditional_eligible", "always_alternative_exit", len(eligible), float(alt.mean()),
            float((oracle - alt).mean()), "OBSERVED", "Clean gross counterfactual; no realised row cost in target.")
        add("conditional_eligible", "hindsight_oracle", len(eligible), float(oracle.mean()), 0.0,
            "ORACLE_DIAGNOSTIC_ONLY", "Uses future outcomes and is not deployable.")
        add("conditional_eligible", "learned_action", len(eligible), None, None,
            "NOT_RUN_TWO_HEAD_SCOPE", "Action head disabled by the approved base-plus-residual architecture.")
    else:
        for policy in ("always_baseline_continue", "always_alternative_exit", "hindsight_oracle"):
            add("conditional_eligible", policy, 0, None, None, "NOT_RUN_NO_ELIGIBLE_ROWS", "No clean action counterfactual rows.")
        add("conditional_eligible", "learned_action", 0, None, None,
            "NOT_RUN_TWO_HEAD_SCOPE", "Action head disabled by the approved base-plus-residual architecture.")

    base_gross = float(ledger.execution_adjusted_gross_h12_bps.mean())
    base_net = float(ledger.net_h12_bps.mean())
    add("complete_upstream_population", "unchanged_entry_baseline_gross", len(ledger), base_gross, None,
        "OBSERVED", "All original candidates; no conditional-row substitution.")
    add("complete_upstream_population", "unchanged_entry_baseline_net", len(ledger), base_net, None,
        "OBSERVED_FUTURE_RESOLVED_FEE", "All original candidates; realised diagnostic net.")
    add("complete_upstream_population", "learned_action_overlay", len(ledger), None, None,
        "NOT_RUN_TWO_HEAD_SCOPE", "No action head is trained or promoted.")
    # Preserve every requested policy-control gate in the machine-readable
    # artifact even though the approved architecture intentionally has no
    # trainable action head.  This prevents an omitted test from looking like
    # a passed test in downstream summaries.
    controls = (
        ("incremental_target_definition", "VERIFIED_CLEAN_GROSS_ARMS", "action delta is alternative gross minus baseline gross; realised row costs excluded"),
        ("independent_prefix_recomputation", "NOT_RUN_TWO_HEAD_SCOPE", "no learned action feature matrix is constructed"),
        ("target_proximity_ablation", "NOT_RUN_TWO_HEAD_SCOPE", "Stage-0 scanner exists, but no action model is fit to ablate"),
        ("leave_target_adjacent_features_out", "NOT_RUN_TWO_HEAD_SCOPE", "no action model is fit"),
        ("latency_slippage_sensitivity", "NOT_RUN_SOURCE_UNAVAILABLE", "canonical identical-row delayed-entry tails are unavailable"),
        ("later_sealed_oos", "NOT_RUN_TWO_HEAD_SCOPE", "no learned action candidate exists to seal"),
        ("identical_entry_complete_population_overlay", "BASELINE_ONLY", "complete unchanged entry population is reported above; no action overlay is applied"),
    )
    for policy, status, detail in controls:
        add("policy_control_gate", policy, len(ledger), None, None, status, detail)
    return pd.DataFrame(rows)


def run(ledger_path: Path, output: Path) -> dict:
    ledger = pd.read_parquet(ledger_path)
    required = {
        "candidate_id", "execution_adjusted_gross_h12_bps", "fee_bps", "net_h12_bps",
        "fixed_ex_ante_fee_bps", "score_base_alpha", "score_residual_alpha", "action_target_status",
        "action_continue_execution_adjusted_gross_bps", "action_exit_execution_adjusted_gross_bps",
    }
    missing = required - set(ledger)
    if missing:
        raise ValueError(f"missing required ledger columns: {sorted(missing)}")
    if ledger.candidate_id.duplicated().any():
        raise ValueError("candidate_id is not unique")
    if not np.allclose(
        ledger.execution_adjusted_gross_h12_bps - ledger.fee_bps,
        ledger.net_h12_bps,
        atol=1e-8,
    ):
        raise ValueError("gross-fee-net reconciliation failed")

    output.mkdir(parents=True, exist_ok=True)
    waterfall = execution_waterfall(ledger)
    regret = policy_regret(ledger)
    waterfall.to_parquet(output / "execution_waterfall.parquet", index=False)
    regret.to_parquet(output / "policy_regret.parquet", index=False)
    checks = {
        "candidate_ids_unique": True,
        "gross_fee_net_reconciles": True,
        "global_not_timestamp_topk": True,
        "missing_counterfactuals_not_imputed": bool(waterfall.status.str.startswith("NOT_").any()),
        "base_and_residual_reported_separately": set(waterfall.loc[waterfall.record_type.eq("global_top10_observed"), "score"]) == {"score_base_alpha", "score_residual_alpha"},
        "action_head_disabled": bool(regret.loc[regret.policy.str.contains("learned_action"), "status"].eq("NOT_RUN_TWO_HEAD_SCOPE").all()),
        "conditional_and_complete_populations_separate": {"conditional_eligible", "complete_upstream_population"}.issubset(set(regret.population)),
        "policy_control_gates_explicit": {
            "independent_prefix_recomputation", "target_proximity_ablation",
            "leave_target_adjacent_features_out", "latency_slippage_sensitivity",
            "later_sealed_oos", "identical_entry_complete_population_overlay",
        }.issubset(set(regret.loc[regret.population.eq("policy_control_gate"), "policy"])),
    }
    manifest = {
        "schema": "root_cause_execution_policy_audit_v1",
        "status": "COMPLETE_RESEARCH_ONLY",
        "promotion_eligible": False,
        "runner": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "architecture": ["base_directional_alpha", "stopped_gradient_residual"],
        "input": str(ledger_path.resolve()),
        "input_sha256": _sha256(ledger_path),
        "rows": len(ledger),
        "ordered_candidate_id_sha256": _id_sha(ledger),
        "selection_semantics": "GLOBAL_TOP_K_NOT_PER_TIMESTAMP",
        "outputs": {},
        "checks": checks,
    }
    for name in ("execution_waterfall.parquet", "policy_regret.parquet"):
        manifest["outputs"][name] = _sha256(output / name)
    (output / "correctness_test_report.json").write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n")
    manifest["outputs"]["correctness_test_report.json"] = _sha256(output / "correctness_test_report.json")
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if not all(checks.values()):
        raise AssertionError(f"correctness checks failed: {checks}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.ledger, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
