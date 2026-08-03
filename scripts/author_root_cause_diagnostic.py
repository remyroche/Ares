#!/usr/bin/env python3
"""Evidence-driven authoring of the final root-cause report and waterfall.

This is a deterministic authoring step, not a model or policy runner. Every
terminal class is marked SUPPORTED only under an explicit roadmap rule tied to
validated source rows; all other classes remain UNRESOLVED. The runner must be
invoked only after canonical Stage3 and global-learning artifacts exist.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from scripts.assemble_root_cause_final_pack import (
    ART, DEFAULT_POINTER, DEFAULT_STAGE2, DEFAULT_STAGE3, DEFAULT_STAGE56,
    EXPECTED_STAGE3_SCOPE, EXPECTED_TWO_HEAD_ARCHITECTURE,
    ROOT, _read_json, _require, _resolve, _verify_ledger_input,
    _verify_manifest_output, _verify_runner, sha256,
)
from scripts.generate_root_cause_diagnostic_scaffold import (
    DEFAULT_GLOBAL_LEARNING, TERMINAL_CLASSES, _require_global,
)


DEFAULT_OUTPUT = ART / "root_cause_diagnostic_authoring_20260731_v1"
BROAD_TOP_FRACTION = 0.20
MODEL_TOP_FRACTION = 0.10


def _atomic_directory(output: Path) -> Path:
    staging = output.with_name(f".{output.name}.staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=False)
    return staging


def _one(frame: pd.DataFrame, mask: pd.Series, description: str) -> pd.Series:
    result = frame.loc[mask]
    _require(len(result) == 1, f"expected exactly one evidence row for {description}; found {len(result)}")
    return result.iloc[0]


def _oracle_row(frame: pd.DataFrame, oracle: str, fraction: float) -> pd.Series:
    return _one(
        frame,
        frame.oracle.eq(oracle) & frame.top_fraction.eq(fraction) & frame.slice_kind.eq("pooled") & frame.slice_value.eq("ALL"),
        f"{oracle} pooled top {fraction}",
    )


def _named_gap(frame: pd.DataFrame, name: str) -> pd.Series:
    return _one(frame, frame.record_type.eq("named_global_gap") & frame.comparison.eq(name), f"global-learning gap {name}")


def _classification(name: str, supported: bool, rule: str, evidence: str, value_bps: float | None = None) -> dict[str, Any]:
    return {
        "classification": name,
        "status": "SUPPORTED" if supported else "UNRESOLVED",
        "rule": rule,
        "evidence": evidence,
        "value_bps": value_bps,
    }


def classify(
    *, oracle_results: pd.DataFrame, global_gaps: pd.DataFrame, global_concordance: pd.DataFrame,
    execution_waterfall: pd.DataFrame, policy_regret: pd.DataFrame,
) -> pd.DataFrame:
    """Apply only explicit roadmap rules; absence of proof remains unresolved."""
    o1 = _oracle_row(oracle_results, "O1_realised_gross_h12", BROAD_TOP_FRACTION)
    o2 = _oracle_row(oracle_results, "O2_realised_net_h12", BROAD_TOP_FRACTION)
    current = _oracle_row(oracle_results, "CURRENT_base_plus_residual_delta_OOF", MODEL_TOP_FRACTION)
    causal_to_future = _named_gap(global_gaps, "causal_to_future")
    production_to_causal = _named_gap(global_gaps, "production_to_causal")

    gross = float(o1.mean_evaluation_gross_bps)
    net = float(o2.mean_net_bps)
    model_net = float(current.mean_net_bps)
    future_gap = float(causal_to_future.net_gap_bps)
    learning_gap = float(production_to_causal.net_gap_bps)
    cost_drag_rows = execution_waterfall.loc[
        execution_waterfall.stage.eq("cost_drag_D_minus_E")
        & execution_waterfall.status.eq("IDENTIFIED")
        & execution_waterfall.value_bps_per_candidate.notna()
    ]
    cost_drag = float(cost_drag_rows.iloc[0].value_bps_per_candidate) if len(cost_drag_rows) == 1 else np.nan

    metric_rows = global_concordance.loc[
        global_concordance.later_global_economic_metric.eq("net_bps")
        & global_concordance.development_base_metric.isin(
            ["base_directional__spearman_ic", "base_directional__roc_auc", "base_directional__pr_auc"]
        )
        & global_concordance.arms.ge(3)
    ].copy()
    metric_rows = metric_rows.loc[np.isfinite(metric_rows.spearman)]
    metric_misaligned = bool((metric_rows.spearman <= 0.0).any())
    metric_evidence = "no finite global metric-concordance row with at least three arms"
    if metric_misaligned:
        row = metric_rows.loc[metric_rows.spearman.le(0.0)].sort_values("spearman").iloc[0]
        metric_evidence = f"{row.development_base_metric} to net_bps Spearman={float(row.spearman):.6f} across {int(row.arms)} arms"

    transfer = execution_waterfall.loc[
        execution_waterfall.stage.isin(
            ["entry_transfer_loss_A_minus_B", "delay_slippage_loss_B_minus_C", "policy_geometry_loss_C_minus_D"]
        )
        & execution_waterfall.value_bps_per_candidate.notna()
        & execution_waterfall.status.isin(["IDENTIFIED", "OBSERVED"])
    ]
    execution_supported = bool((transfer.value_bps_per_candidate < 0.0).any())
    execution_evidence = "entry-transfer, delay/slippage, and geometry losses are not identified from sealed source rows"
    execution_value = np.nan
    if execution_supported:
        row = transfer.sort_values("value_bps_per_candidate").iloc[0]
        execution_evidence = f"{row.stage}={float(row.value_bps_per_candidate):.6f} bps"
        execution_value = float(row.value_bps_per_candidate)

    learned = policy_regret.loc[
        policy_regret.population.eq("complete_upstream_population")
        & policy_regret.policy.eq("learned_action_overlay")
        & policy_regret.status.eq("OBSERVED")
        & policy_regret.oracle_regret_bps_per_candidate.notna()
    ]
    policy_supported = bool((learned.oracle_regret_bps_per_candidate > 0.0).any())
    policy_evidence = "no learned action overlay with observed complete-population policy regret"
    policy_value = np.nan
    if policy_supported:
        row = learned.iloc[0]
        policy_evidence = f"complete-population learned-action regret={float(row.oracle_regret_bps_per_candidate):.6f} bps"
        policy_value = float(row.oracle_regret_bps_per_candidate)

    rows = [
        _classification(
            "TARGET_OR_POPULATION_FAILURE", gross < 0.0,
            "broad realised gross oracle at global top-20 is negative",
            f"O1 realised gross top-20={gross:.6f} bps", gross,
        ),
        _classification(
            "COST_DRAG_FAILURE", gross > 0.0 and net < 0.0 and np.isfinite(cost_drag) and cost_drag > 0.0,
            "broad O1 gross tail is positive, broad O2 net tail is negative, and observed cost drag is positive",
            f"O1 gross top-20={gross:.6f}; O2 net top-20={net:.6f}; observed full-population cost drag={cost_drag:.6f} bps",
            cost_drag,
        ),
        _classification(
            "CAUSAL_FEATURE_INFORMATION_INSUFFICIENT", model_net <= 0.0 and future_gap > 0.0,
            "current global top-10 net is non-positive and future-feature minus causal net gap is positive",
            f"current net top-10={model_net:.6f}; causal-to-future net gap={future_gap:.6f} bps", future_gap,
        ),
        _classification(
            "ML_LEARNING_EFFICIENCY_FAILURE", learning_gap > 0.0,
            "causal-capacity oracle exceeds production-like model on later global top-10 net",
            f"production-to-causal net gap={learning_gap:.6f} bps", learning_gap,
        ),
        _classification(
            "METRIC_SELECTION_MISALIGNMENT", metric_misaligned,
            "a development metric has non-positive rank association with later global top-10 net across at least three arms",
            metric_evidence, np.nan,
        ),
        _classification(
            "EXECUTION_TRANSFER_FAILURE", execution_supported,
            "an identified entry-transfer, delay/slippage, or geometry loss is negative",
            execution_evidence, execution_value,
        ),
        _classification(
            "POLICY_CONVERSION_FAILURE", policy_supported,
            "learned action has observed positive complete-population regret versus hindsight policy",
            policy_evidence, policy_value,
        ),
    ]
    # Exactly one row per roadmap terminal class; this guard makes coverage
    # drift fail before report generation.
    out = pd.DataFrame(rows)
    _require(set(out.classification) == set(TERMINAL_CLASSES), "classification rules do not cover the roadmap terminal classes exactly once")
    out["economic_rank"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    rankable = out.status.eq("SUPPORTED") & out.value_bps.notna()
    ordered = out.loc[rankable].assign(__absolute_bps=out.loc[rankable, "value_bps"].abs()).sort_values("__absolute_bps", ascending=False, kind="stable")
    out.loc[ordered.index, "economic_rank"] = range(1, len(ordered) + 1)
    return out.sort_values("classification", kind="stable").reset_index(drop=True)


def build_waterfall(
    *, execution_waterfall: pd.DataFrame, oracle_results: pd.DataFrame, oracle_regret: pd.DataFrame,
    global_arms: pd.DataFrame, global_gaps: pd.DataFrame, classifications: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in execution_waterfall.iterrows():
        rows.append({
            "record_type": "execution_waterfall",
            "component": str(row.stage),
            "metric": "value_bps_per_candidate",
            "value_bps": row.value_bps_per_candidate,
            "status": row.status,
            "source_artifact": "stage56_execution_waterfall",
            "source_selector": f"record_type={row.record_type};slice={row.slice};score={row.score}",
            "detail": row.detail,
        })
    for oracle in ("O1_realised_gross_h12", "O2_realised_net_h12", "CURRENT_base_plus_residual_delta_OOF"):
        for fraction in (0.01, 0.05, 0.10, 0.20):
            row = _oracle_row(oracle_results, oracle, fraction)
            metric = "mean_net_bps" if oracle == "O2_realised_net_h12" else "mean_evaluation_gross_bps"
            rows.append({
                "record_type": "oracle_tail",
                "component": f"{oracle}_global_top_{int(fraction * 100)}",
                "metric": metric,
                "value_bps": float(row[metric]),
                "status": row.net_status if metric == "mean_net_bps" else "AVAILABLE",
                "source_artifact": "stage1_oracle_ladder_results",
                "source_selector": f"oracle={oracle};top_fraction={fraction};slice=pooled/ALL",
                "detail": "Exact global top-k, never per timestamp or side.",
            })
    for _, row in oracle_regret.iterrows():
        rows.append({
            "record_type": "oracle_regret",
            "component": f"{row.oracle}_global_top_{int(float(row.top_fraction) * 100)}",
            "metric": "entry_regret_bps",
            "value_bps": row.entry_regret_bps,
            "status": row.oracle_net_status,
            "source_artifact": "stage1_oracle_regret",
            "source_selector": f"oracle={row.oracle};top_fraction={row.top_fraction}",
            "detail": "Regret against the frozen current OOF reference.",
        })
    for _, row in global_arms.iterrows():
        if float(row.top_fraction) != MODEL_TOP_FRACTION:
            continue
        rows.append({
            "record_type": "global_learning_arm",
            "component": str(row.model_family),
            "metric": "net_bps",
            "value_bps": row.net_bps,
            "status": "AVAILABLE_GLOBAL_TOP_K",
            "source_artifact": "global_topk_learning_economics",
            "source_selector": f"model_family={row.model_family};seed={row.seed};top_fraction={row.top_fraction}",
            "detail": row.selection_scope,
        })
    for _, row in global_gaps.loc[global_gaps.record_type.eq("named_global_gap")].iterrows():
        rows.append({
            "record_type": "global_learning_gap",
            "component": str(row.comparison),
            "metric": "net_gap_bps",
            "value_bps": row.net_gap_bps,
            "status": "AVAILABLE_GLOBAL_TOP_K",
            "source_artifact": "global_topk_learning_gaps",
            "source_selector": f"comparison={row.comparison}",
            "detail": row.selection_scope,
        })
    for _, row in classifications.iterrows():
        rows.append({
            "record_type": "classification",
            "component": row.classification,
            "metric": "supporting_value_bps",
            "value_bps": row.value_bps,
            "status": row.status,
            "source_artifact": "derived_from_sealed_evidence",
            "source_selector": row.rule,
            "detail": row.evidence,
            "economic_rank": row.economic_rank,
        })
    return pd.DataFrame(rows)


def _render_report(
    *, classifications: pd.DataFrame, feature_results: pd.DataFrame, source_hashes: Mapping[str, str],
    waterfall: pd.DataFrame, head_metrics: pd.DataFrame,
) -> str:
    class_rows = "\n".join(
        f"| {row.classification} | {row.status} | {row.rule} | {row.evidence} |"
        for _, row in classifications.iterrows()
    )
    supported = waterfall.loc[
        waterfall.record_type.eq("classification") & waterfall.status.eq("SUPPORTED") & waterfall.economic_rank.notna()
    ].sort_values("economic_rank", kind="stable")
    ranking = "\n".join(
        f"| {int(row.economic_rank)} | {row.component} | {row.value_bps:.6f} | {row.detail} |" for _, row in supported.iterrows()
    ) or "| n/a | None | n/a | No supported class has a causally comparable bps contribution. |"
    feature_summary = (
        f"Feature rows: {len(feature_results)}; unique fields: {feature_results.feature_name.nunique()}; "
        f"side-local max transported IC: {float(feature_results.transported_ic_mean.max()):.6f}; "
        f"max top-bottom decile spread: {float(feature_results.top_bottom_decile_spread_mean_bps.max()):.6f} bps."
    )
    # Keep the two approved heads economically and statistically separate in
    # the report.  The residual head is not scored with directional metrics,
    # and the directional head is not scored with economic residual metrics.
    if {"evaluation_scope", "split", "component"}.issubset(head_metrics.columns):
        held = head_metrics.loc[
            head_metrics.evaluation_scope.eq("outer_heldout")
            & head_metrics.split.eq("later_oos")
        ].copy()
    else:
        # Synthetic contract tests and partial evidence must remain readable;
        # absence of head metrics is reported as n/a rather than inferred.
        held = pd.DataFrame(columns=["model_family", "side", "component"])
    base_metrics = (
        "base_directional__spearman_ic", "base_directional__roc_auc",
        "base_directional__pr_auc", "base_directional__log_loss",
        "base_directional__brier", "base_directional__ece",
        "base_directional__mae",
    )
    residual_metrics = (
        "residual_economic__spearman_ic", "residual_economic__mae_bps",
        "residual_economic__huber_bps", "residual_economic__gross_mean_bps",
        "residual_economic__net_mean_bps", "residual_economic__gross_top10_bps",
        "residual_economic__net_top10_bps", "residual_economic__gross_top20_bps",
        "residual_economic__net_top20_bps",
    )

    def _metric_table(component: str, names: tuple[str, ...]) -> str:
        part = held.loc[held.component.eq(component)].copy()
        if part.empty:
            return "| n/a | n/a | No later-OOS rows. |"
        rows: list[str] = []
        for (family, side), group in part.groupby(["model_family", "side"], observed=True):
            values = []
            for name in names:
                value = pd.to_numeric(group[name], errors="coerce").mean() if name in group else np.nan
                values.append("n/a" if not np.isfinite(value) else f"{float(value):.6f}")
            rows.append("| " + " | ".join([str(family), str(side), *values]) + " |")
        return "\n".join(rows) or "| n/a | n/a | No later-OOS rows. |"

    base_header = "| Model family | Side | " + " | ".join(base_metrics) + " |"
    base_sep = "|---|---|" + "---|" * len(base_metrics)
    residual_header = "| Model family | Side | " + " | ".join(residual_metrics) + " |"
    residual_sep = "|---|---|" + "---|" * len(residual_metrics)
    base_table = "\n".join([base_header, base_sep, _metric_table("base_directional", base_metrics)])
    residual_table = "\n".join([residual_header, residual_sep, _metric_table("residual_economic", residual_metrics)])
    hashes = "\n".join(f"- {name}: {digest}" for name, digest in source_hashes.items())
    return f"""# Root-Cause Diagnostic Report

Status: EVIDENCE_DRIVEN_DIAGNOSTIC_ONLY — NO PROMOTION DECISION

All classifications below use explicit rules embedded in the sealed authoring
runner. A class is SUPPORTED only when its rule is true on validated evidence;
otherwise it is UNRESOLVED. Global top-k results are never timestamp-local.

## Terminal decomposition

| Classification | Status | Rule | Evidence |
|---|---|---|---|
{class_rows}

## Supported classes ranked by available bps contribution

| Rank | Classification | Supporting bps | Evidence |
|---:|---|---:|---|
{ranking}

Only supported classes with a causally comparable bps value are ranked.
Unidentified execution transfer terms remain unidentified rather than zero.

## Feature-information evidence

{feature_summary}

## Directional base-head metrics (later OOS)

These are directional/soft-alpha metrics only; they are intentionally not
used as residual-economic metrics.

{base_table}

## Stopped-gradient residual-head metrics (later OOS)

These are gross/net economic residual metrics only; they are intentionally not
used as directional classification metrics.

{residual_table}

## Evidence digests

{hashes}
"""


def author(
    *, pointer_path: Path = DEFAULT_POINTER, stage2_dir: Path = DEFAULT_STAGE2,
    stage3_dir: Path = DEFAULT_STAGE3, stage56_dir: Path = DEFAULT_STAGE56,
    global_learning_dir: Path = DEFAULT_GLOBAL_LEARNING, output: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    _require(not output.exists(), f"refusing to overwrite immutable diagnostic: {output}")
    pointer = _read_json(pointer_path)
    stage0_dir = _resolve(pointer.get("stage0_substrate", "__missing__"))
    stage1_dir = _resolve(pointer.get("stage1_oracle_ladder", "__missing__"))
    stage0_manifest_path = stage0_dir / "diagnostic_population_manifest.json"
    stage1_manifest_path = stage1_dir / "run_manifest.json"
    _require(sha256(stage0_manifest_path) == pointer.get("stage0_manifest_sha256"), "canonical pointer Stage0 manifest digest mismatch")
    _require(sha256(stage1_manifest_path) == pointer.get("stage1_manifest_sha256"), "canonical pointer Stage1 manifest digest mismatch")
    stage0 = _read_json(stage0_manifest_path)
    stage1 = _read_json(stage1_manifest_path)
    stage2 = _read_json(stage2_dir / "run_manifest.json")
    stage3_manifest_path = stage3_dir / "run_manifest.json"
    stage3 = _read_json(stage3_manifest_path)
    stage56 = _read_json(stage56_dir / "run_manifest.json")
    ledger = _verify_manifest_output(stage0_dir, stage0, "diagnostic_row_ledger.parquet", "outputs_sha256")
    ledger_sha = sha256(ledger)
    runners = {
        "stage0": _verify_runner(stage0, name="Stage0"),
        "stage1": _verify_runner(stage1, name="Stage1"),
        "stage2": _verify_runner(stage2, name="Stage2", legacy_code_key="code_sha256", expected_path=ROOT / "scripts/run_root_cause_feature_information_audit.py"),
        "stage3": _verify_runner(stage3, name="Stage3"),
        "stage56": _verify_runner(stage56, name="Stage5/6"),
    }
    for name, manifest in (("stage1", stage1), ("stage2", stage2), ("stage3", stage3), ("stage56", stage56)):
        _verify_ledger_input(manifest, name=name, ledger=ledger, ledger_sha=ledger_sha)
    _require(stage3.get("scope") == EXPECTED_STAGE3_SCOPE, "Stage3 scope is not approved two-head scope")
    _require(stage3.get("invariants", {}).get("no_auxiliary_or_policy_layers") is True, "Stage3 admits an auxiliary or policy layer")
    architecture = tuple(stage56.get("architecture", ()))
    _require(architecture == EXPECTED_TWO_HEAD_ARCHITECTURE, f"Stage5/6 violates two-head architecture: {architecture}")
    _require(stage56.get("checks", {}).get("action_head_disabled") is True, "Stage5/6 action-head disable gate failed")
    global_manifest, global_outputs = _require_global(global_learning_dir, stage3_manifest_path)
    runners["global_learning"] = _verify_runner(global_manifest, name="Global-learning")

    sources = {
        "oracle_ladder_results": _verify_manifest_output(stage1_dir, stage1, "oracle_ladder_results.parquet", "outputs_sha256"),
        "oracle_regret": _verify_manifest_output(stage1_dir, stage1, "oracle_regret_vs_current_oof.parquet", "outputs_sha256"),
        "feature_information_results": _verify_manifest_output(stage2_dir, stage2, "feature_information_results.parquet", "outputs_sha256"),
        "model_learning_efficiency": _verify_manifest_output(stage3_dir, stage3, "model_learning_efficiency.parquet", "outputs_sha256"),
        "metric_concordance": _verify_manifest_output(stage3_dir, stage3, "metric_concordance.parquet", "outputs_sha256"),
        "execution_waterfall": _verify_manifest_output(stage56_dir, stage56, "execution_waterfall.parquet", "outputs"),
        "policy_regret": _verify_manifest_output(stage56_dir, stage56, "policy_regret.parquet", "outputs"),
        "global_learning_economics": global_outputs["global_topk_learning_economics.parquet"],
        "global_learning_gaps": global_outputs["global_topk_learning_gaps.parquet"],
        "global_metric_concordance": global_outputs["causal_only_global_metric_concordance.parquet"],
    }
    source_hashes = {name: sha256(path) for name, path in sources.items()}
    data = {name: pd.read_parquet(path) for name, path in sources.items()}
    classifications = classify(
        oracle_results=data["oracle_ladder_results"], global_gaps=data["global_learning_gaps"],
        global_concordance=data["global_metric_concordance"], execution_waterfall=data["execution_waterfall"],
        policy_regret=data["policy_regret"],
    )
    waterfall = build_waterfall(
        execution_waterfall=data["execution_waterfall"], oracle_results=data["oracle_ladder_results"],
        oracle_regret=data["oracle_regret"], global_arms=data["global_learning_economics"],
        global_gaps=data["global_learning_gaps"], classifications=classifications,
    )
    stage = _atomic_directory(output)
    try:
        classifications.to_parquet(stage / "classification_evidence.parquet", index=False)
        waterfall.to_parquet(stage / "root_cause_waterfall.parquet", index=False)
        (stage / "ROOT_CAUSE_DIAGNOSTIC_REPORT.md").write_text(
            _render_report(
                classifications=classifications,
                feature_results=data["feature_information_results"],
                source_hashes=source_hashes,
                waterfall=waterfall,
                head_metrics=data["model_learning_efficiency"],
            )
        )
        outputs = {name: sha256(stage / name) for name in ("classification_evidence.parquet", "root_cause_waterfall.parquet", "ROOT_CAUSE_DIAGNOSTIC_REPORT.md")}
        correctness = {
            "schema": "root_cause_diagnostic_authoring_correctness_v1",
            "status": "PASS_EVIDENCE_DRIVEN_DIAGNOSTIC_ONLY",
            "ledger_sha256": ledger_sha,
            "runners": runners,
            "source_hashes": source_hashes,
            "two_head_scope": {"stage3_scope": stage3["scope"], "stage56_architecture": list(architecture)},
            "classifications": classifications.to_dict(orient="records"),
            "outputs_sha256": outputs,
        }
        (stage / "correctness_test_report.json").write_text(json.dumps(correctness, indent=2, sort_keys=True) + "\n")
        outputs["correctness_test_report.json"] = sha256(stage / "correctness_test_report.json")
        manifest = {
            **correctness,
            "schema": "root_cause_diagnostic_authoring_v1",
            "status": "COMPLETE_EVIDENCE_DRIVEN_DIAGNOSTIC_ONLY_NO_PROMOTION",
            "runner": {"path": str(Path(__file__).resolve().relative_to(ROOT)), "sha256": sha256(Path(__file__))},
            "source_manifest_hashes": {
                "pointer": sha256(pointer_path), "stage0": sha256(stage0_manifest_path), "stage1": sha256(stage1_manifest_path),
                "stage2": sha256(stage2_dir / "run_manifest.json"), "stage3": sha256(stage3_manifest_path),
                "stage56": sha256(stage56_dir / "run_manifest.json"), "global_learning": sha256(global_learning_dir / "run_manifest.json"),
            },
        }
        (stage / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        (stage / "manifest.sha256").write_text(sha256(stage / "run_manifest.json") + "\n")
        stage.rename(output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pointer", type=Path, default=DEFAULT_POINTER)
    parser.add_argument("--stage2", type=Path, default=DEFAULT_STAGE2)
    parser.add_argument("--stage3", type=Path, default=DEFAULT_STAGE3)
    parser.add_argument("--stage56", type=Path, default=DEFAULT_STAGE56)
    parser.add_argument("--global-learning", type=Path, default=DEFAULT_GLOBAL_LEARNING)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(author(pointer_path=args.pointer, stage2_dir=args.stage2, stage3_dir=args.stage3, stage56_dir=args.stage56, global_learning_dir=args.global_learning, output=args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
