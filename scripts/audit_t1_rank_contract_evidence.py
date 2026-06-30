#!/usr/bin/env python3
"""Audit evidence for the provisional T1 short_boll rank contract.

This script consumes already-materialized rank-contract validation artifacts.
It does not rerun replay and it does not change the active production stack.

The purpose is to make the promotion state explicit:

* pre-June walk-forward can support the global-over-time challenger;
* later fixed-contract blocks can support or contradict that challenger;
* conflicting evidence keeps the measured timestamp-rank T1 baseline
  provisional rather than silently promoting either contract.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TIMESTAMP_ARM = "timestamp_rank_t1"
GLOBAL_ARM = "fold_causal_global_rank_reference"
BASELINE_STATIC_ARM = "S0_baseline_static_thresholds"


@dataclass(frozen=True)
class EvidenceThresholds:
    min_prejune_folds: int = 3
    min_later_timestamps: int = 24
    min_later_base_trades: int = 30
    min_later_challenger_trades: int = 30
    min_positive_fold_share: float = 0.60


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def _metric_delta(delta: pd.DataFrame, metric: str) -> float:
    if delta.empty or not {"metric", "challenger_minus_base"}.issubset(delta.columns):
        return float("nan")
    rows = delta.loc[delta["metric"].astype(str).eq(metric), "challenger_minus_base"]
    if rows.empty:
        return float("nan")
    return _finite_float(rows.iloc[0])


def _support_from_delta(delta_net_pnl: float, *, tolerance: float = 1e-12) -> str:
    if not np.isfinite(delta_net_pnl) or abs(delta_net_pnl) <= tolerance:
        return "tie_or_unknown"
    return "global_rank" if delta_net_pnl > 0 else "timestamp_rank"


def audit_prejune_walkforward(walkforward_dir: Path, thresholds: EvidenceThresholds) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    manifest_path = walkforward_dir / "t1_rank_contract_walkforward_manifest.json"
    aggregate_path = walkforward_dir / "rank_contract_walkforward_aggregate.csv"
    fold_delta_path = walkforward_dir / "rank_contract_walkforward_fold_delta.csv"
    overlap_path = walkforward_dir / "rank_contract_walkforward_accepted_overlap.csv"
    rank_diag_path = walkforward_dir / "rank_contract_walkforward_rank_diagnostics.csv"
    required = [manifest_path, aggregate_path, fold_delta_path, overlap_path, rank_diag_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        return {"walkforward_dir": str(walkforward_dir), "support": "missing"}, [f"missing pre-June artifacts: {missing}"]

    manifest = _load_json(manifest_path)
    aggregate = _read_csv(aggregate_path)
    fold_delta = _read_csv(fold_delta_path)
    overlap = _read_csv(overlap_path)
    rank_diag = _read_csv(rank_diag_path)

    if manifest.get("generated_by") != "run_t1_rank_contract_walkforward":
        failures.append("pre-June manifest generated_by is unexpected")
    if manifest.get("purpose") != "pre_june_rank_contract_validation":
        failures.append("pre-June manifest purpose is unexpected")
    failures.extend(_audit_fixed_policy_contract(manifest.get("fixed_policy_contract"), prefix="pre-June fixed_policy_contract"))
    leakage = manifest.get("leakage_contract")
    if not isinstance(leakage, dict):
        failures.append("pre-June manifest leakage_contract is missing")
    else:
        expected = {
            "split_by_complete_timestamps": True,
            "global_rank_reference_uses_validation_rows": False,
            "global_rank_reference_uses_future_rows": False,
            "market_state_controller_active": False,
            "qfail_active": False,
            "rank_contract_is_the_only_arm_difference": True,
        }
        for key, value in expected.items():
            if leakage.get(key) is not value:
                failures.append(f"pre-June leakage_contract.{key} != {value}")

    arms = manifest.get("arms") if isinstance(manifest.get("arms"), dict) else {}
    if arms.get(TIMESTAMP_ARM, {}).get("rank_scope") != "within_timestamp":
        failures.append("pre-June timestamp arm rank_scope is not within_timestamp")
    if arms.get(GLOBAL_ARM, {}).get("rank_scope") != "global_over_time":
        failures.append("pre-June global arm rank_scope is not global_over_time")
    if arms.get(GLOBAL_ARM, {}).get("fit_scope") != "training_timestamps_only_per_fold":
        failures.append("pre-June global arm fit_scope is not training_timestamps_only_per_fold")

    if not {"arm", "total_net_pnl", "total_trades", "folds"}.issubset(aggregate.columns):
        failures.append("pre-June aggregate missing required columns")
    if not {"fold", "delta_net_pnl", "accepted_jaccard"}.issubset(fold_delta.columns):
        failures.append("pre-June fold_delta missing required columns")
    if not {"fold", "jaccard", "removed_net_pnl", "added_net_pnl"}.issubset(overlap.columns):
        failures.append("pre-June accepted_overlap missing required columns")

    fold_count = int(fold_delta["fold"].nunique()) if "fold" in fold_delta.columns else 0
    if fold_count < thresholds.min_prejune_folds:
        failures.append(f"pre-June fold_count {fold_count} < {thresholds.min_prejune_folds}")
    if int(manifest.get("fold_count") or 0) != fold_count:
        failures.append("pre-June manifest fold_count does not match fold_delta")

    rank_diag_required = [
        "global_valid_missing_policy_rank_rows",
        "global_valid_missing_auction_rank_rows",
        "global_train_missing_policy_rank_rows",
        "global_train_missing_auction_rank_rows",
        "global_valid_ranked_rows",
        "valid_broad_rows",
        "global_train_ranked_rows",
        "train_deployable_rows",
    ]
    if not set(rank_diag_required).issubset(rank_diag.columns):
        failures.append("pre-June rank diagnostics missing required columns")
    else:
        for column in [
            "global_valid_missing_policy_rank_rows",
            "global_valid_missing_auction_rank_rows",
            "global_train_missing_policy_rank_rows",
            "global_train_missing_auction_rank_rows",
        ]:
            if int(pd.to_numeric(rank_diag[column], errors="coerce").fillna(0).sum()) != 0:
                failures.append(f"pre-June rank diagnostics has missing ranks in {column}")
        valid_gap = (
            pd.to_numeric(rank_diag["global_valid_ranked_rows"], errors="coerce")
            - pd.to_numeric(rank_diag["valid_broad_rows"], errors="coerce")
        ).abs()
        train_gap = (
            pd.to_numeric(rank_diag["global_train_ranked_rows"], errors="coerce")
            - pd.to_numeric(rank_diag["train_deployable_rows"], errors="coerce")
        ).abs()
        if bool((valid_gap > 0).any()):
            failures.append("pre-June global valid ranked rows do not equal valid broad rows")
        if bool((train_gap > 0).any()):
            failures.append("pre-June global train ranked rows do not equal train deployable rows")

    timestamp_rows = aggregate.loc[aggregate.get("arm", pd.Series(dtype=object)).astype(str).eq(TIMESTAMP_ARM)]
    global_rows = aggregate.loc[aggregate.get("arm", pd.Series(dtype=object)).astype(str).eq(GLOBAL_ARM)]
    timestamp_total = _finite_float(timestamp_rows["total_net_pnl"].iloc[0]) if not timestamp_rows.empty else float("nan")
    global_total = _finite_float(global_rows["total_net_pnl"].iloc[0]) if not global_rows.empty else float("nan")
    delta_total = global_total - timestamp_total
    deltas = pd.to_numeric(fold_delta.get("delta_net_pnl"), errors="coerce")
    positive_share = float((deltas > 0).mean()) if len(deltas) else float("nan")
    median_delta = _finite_float(deltas.median())
    q25_delta = _finite_float(deltas.quantile(0.25))
    support = _support_from_delta(delta_total)
    if support == "global_rank" and (
        not np.isfinite(median_delta)
        or median_delta <= 0
        or not np.isfinite(positive_share)
        or positive_share < thresholds.min_positive_fold_share
    ):
        support = "mixed_or_weak_global_rank"

    return (
        {
            "walkforward_dir": str(walkforward_dir),
            "fold_count": fold_count,
            "timestamp_total_net_pnl": timestamp_total,
            "global_total_net_pnl": global_total,
            "global_minus_timestamp_net_pnl": delta_total,
            "median_delta_net_pnl": median_delta,
            "q25_delta_net_pnl": q25_delta,
            "positive_delta_share": positive_share,
            "mean_accepted_jaccard": _finite_float(pd.to_numeric(fold_delta.get("accepted_jaccard"), errors="coerce").mean()),
            "removed_net_pnl_sum": _finite_float(pd.to_numeric(overlap.get("removed_net_pnl"), errors="coerce").sum()),
            "added_net_pnl_sum": _finite_float(pd.to_numeric(overlap.get("added_net_pnl"), errors="coerce").sum()),
            "support": support,
        },
        failures,
    )


def audit_later_comparison(comparison_dir: Path, thresholds: EvidenceThresholds) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    manifest_path = comparison_dir / "rank_contract_comparison_manifest.json"
    summary_path = comparison_dir / "rank_contract_summary.csv"
    delta_path = comparison_dir / "rank_contract_delta.csv"
    overlap_path = comparison_dir / "rank_contract_accepted_overlap.json"
    deployable_path = comparison_dir / "rank_contract_deployable_rows.csv"
    required = [manifest_path, summary_path, delta_path, overlap_path, deployable_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        return {"comparison_dir": str(comparison_dir), "support": "missing"}, [f"missing later comparison artifacts: {missing}"]

    manifest = _load_json(manifest_path)
    summary = _read_csv(summary_path)
    delta = _read_csv(delta_path)
    overlap_payload = _load_json(overlap_path)
    deployable = _read_csv(deployable_path)
    if manifest.get("generated_by") != "compare_t1_rank_contracts":
        failures.append("later comparison manifest generated_by is unexpected")
    if manifest.get("purpose") != "later_fixed_contract_timestamp_vs_global_rank_validation":
        failures.append("later comparison manifest purpose is unexpected")
    validation = manifest.get("validation")
    if not isinstance(validation, dict):
        failures.append("later comparison manifest validation is missing")
    else:
        if validation.get("passed") is not True:
            failures.append("later comparison manifest validation.passed is not true")
        if validation.get("failures") not in ([], None):
            failures.append("later comparison manifest validation.failures is not empty")
        if validation.get("rank_contract_is_the_only_arm_difference") is not True:
            failures.append("later comparison manifest rank_contract_is_the_only_arm_difference is not true")
        if validation.get("score_threshold_auction_ev_costs_fixed") is not True:
            failures.append("later comparison manifest score_threshold_auction_ev_costs_fixed is not true")
        if validation.get("controller_and_qfail_disabled") is not True:
            failures.append("later comparison manifest controller_and_qfail_disabled is not true")
        if "candidate_universe_identical" in validation and validation.get("candidate_universe_identical") is not True:
            failures.append("later comparison manifest candidate_universe_identical is not true")
    base = manifest.get("base") if isinstance(manifest.get("base"), dict) else {}
    challenger = manifest.get("challenger") if isinstance(manifest.get("challenger"), dict) else {}
    if base.get("rank_contract") != "short_boll_timestamp_rank" or base.get("rank_scope") != "within_timestamp":
        failures.append("later comparison base rank contract is not timestamp T1")
    if challenger.get("rank_contract") != "anchor_global_policy_rank_reference" or challenger.get("rank_scope") != "global_over_time":
        failures.append("later comparison challenger rank contract is not global-over-time")
    rank_contract = challenger.get("rank_reference_contract")
    if not isinstance(rank_contract, dict):
        failures.append("later comparison challenger rank_reference_contract is missing")
    else:
        if rank_contract.get("required") is not True:
            failures.append("later comparison challenger rank_reference_contract.required is not true")
        if rank_contract.get("passed") is not True:
            failures.append("later comparison challenger rank_reference_contract.passed is not true")
        if rank_contract.get("failures") not in ([], None):
            failures.append("later comparison challenger rank_reference_contract.failures is not empty")
    rank_diagnostics = challenger.get("rank_reference_diagnostics")
    if not isinstance(rank_diagnostics, dict):
        failures.append("later comparison challenger rank_reference_diagnostics is missing")
    else:
        for split in ("eval", "train_deployable"):
            diag = rank_diagnostics.get(split)
            if not isinstance(diag, dict):
                failures.append(f"later comparison challenger {split} rank_reference_diagnostics is missing")
                continue
            if diag.get("rank_source") != "policy_rank_reference_percentile":
                failures.append(f"later comparison challenger {split} rank_source is not policy reference")
            if int(diag.get("missing_rank_rows") or 0) != 0:
                failures.append(f"later comparison challenger {split} missing policy ranks is nonzero")
            if int(diag.get("missing_auction_rank_rows") or 0) != 0:
                failures.append(f"later comparison challenger {split} missing auction ranks is nonzero")
            if bool(diag.get("window_rank_debug_used")):
                failures.append(f"later comparison challenger {split} used window-rank debug fallback")
    failures.extend(_audit_fixed_policy_contract(manifest.get("fixed_policy_contract"), prefix="later fixed_policy_contract"))
    if len(summary) != 2:
        failures.append("later comparison summary must contain exactly two contracts")
    if "contract_name" not in summary.columns:
        failures.append("later comparison summary missing contract_name")
    if "metric" not in delta.columns or "challenger_minus_base" not in delta.columns:
        failures.append("later comparison delta missing required columns")
    if not isinstance(overlap_payload.get("overlap"), dict) or not isinstance(overlap_payload.get("swap_pnl"), dict):
        failures.append("later comparison overlap JSON missing overlap/swap_pnl")
    if not {"contract_name", "head", "deployable_rows", "timestamp_count"}.issubset(deployable.columns):
        failures.append("later comparison deployable rows missing required columns")

    delta_net_pnl = _metric_delta(delta, "net_pnl")
    support = _support_from_delta(delta_net_pnl)
    overlap = overlap_payload.get("overlap", {}) if isinstance(overlap_payload.get("overlap"), dict) else {}
    swap = overlap_payload.get("swap_pnl", {}) if isinstance(overlap_payload.get("swap_pnl"), dict) else {}
    timestamp_count = int(pd.to_numeric(deployable.get("timestamp_count"), errors="coerce").max()) if "timestamp_count" in deployable.columns else 0
    base_trades = int(overlap.get("base_accepted") or 0)
    challenger_trades = int(overlap.get("challenger_accepted") or 0)
    candidate_universe_payload = manifest.get("candidate_universe")
    candidate_universe_rows = None
    candidate_universe_jaccard = None
    if isinstance(candidate_universe_payload, dict):
        candidate_overlap = candidate_universe_payload.get("overlap")
        if isinstance(candidate_overlap, dict):
            candidate_universe_rows = int(candidate_overlap.get("base_keys") or 0)
            candidate_universe_jaccard = _finite_float(candidate_overlap.get("jaccard"))
    sufficiently_mature = (
        timestamp_count >= thresholds.min_later_timestamps
        and base_trades >= thresholds.min_later_base_trades
        and challenger_trades >= thresholds.min_later_challenger_trades
    )
    status = "matured_later_block" if sufficiently_mature else "informative_small_sample_not_promotion"
    if not set(deployable["head"].dropna().astype(str)).issuperset({"short_asset", "short_boll"}):
        failures.append("later comparison deployable rows missing active heads")

    return (
        {
            "comparison_dir": str(comparison_dir),
            "manifest": str(manifest_path),
            "status": status,
            "timestamp_count": timestamp_count,
            "candidate_universe_rows": candidate_universe_rows,
            "candidate_universe_jaccard": candidate_universe_jaccard,
            "base_accepted": base_trades,
            "challenger_accepted": challenger_trades,
            "global_minus_timestamp_net_pnl": delta_net_pnl,
            "global_minus_timestamp_trade_count": _metric_delta(delta, "trade_count"),
            "global_minus_timestamp_full_sl_rate": _metric_delta(delta, "full_sl_rate"),
            "accepted_jaccard": _finite_float(overlap.get("jaccard")),
            "base_only": int(overlap.get("base_only") or 0),
            "challenger_only": int(overlap.get("challenger_only") or 0),
            "removed_net_pnl": _finite_float(swap.get("removed_net_pnl")),
            "added_net_pnl": _finite_float(swap.get("added_net_pnl")),
            "removed_winner_pnl": _finite_float(swap.get("removed_winner_pnl")),
            "added_loser_loss": _finite_float(swap.get("added_loser_loss")),
            "support": support,
        },
        failures,
    )


def _audit_fixed_policy_contract(payload: Any, *, prefix: str) -> list[str]:
    failures: list[str] = []
    if not isinstance(payload, dict):
        return [f"{prefix} is missing"]
    expected = {
        "score_path": "anchor_meta_calibrated_score",
        "active_score_column": "calibrated_score",
        "static_base_thresholds": True,
        "policy_variant": "refit_bar4_strategy_bar2",
        "active_heads": ["short_asset", "short_boll"],
        "disabled_heads": ["long_bars", "long_dist"],
        "auction": "global_auction",
        "ev_mapping": "hierarchical_ev_curves_fitted_on_train_deployable",
        "market_state_threshold_controller_active": False,
        "qfail_active": False,
        "native_reliability_blend_active": False,
        "rank_contract_is_the_only_arm_difference": True,
    }
    for key, expected_value in expected.items():
        actual = payload.get(key)
        compare_expected = expected_value
        if isinstance(compare_expected, list):
            actual = sorted(map(str, actual or []))
            compare_expected = sorted(map(str, compare_expected))
        if actual != compare_expected:
            failures.append(f"{prefix}.{key} != {expected_value}")
    return failures


def build_rank_contract_evidence(
    *,
    pre_june_walkforward_dir: Path,
    later_comparison_dirs: list[Path],
    thresholds: EvidenceThresholds,
) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    pre_june, pre_failures = audit_prejune_walkforward(pre_june_walkforward_dir, thresholds)
    failures.extend(pre_failures)
    later_blocks: list[dict[str, Any]] = []
    for comparison_dir in later_comparison_dirs:
        block, block_failures = audit_later_comparison(comparison_dir, thresholds)
        later_blocks.append(block)
        failures.extend(block_failures)

    later_supports = [str(block.get("support")) for block in later_blocks]
    matured_later = [block for block in later_blocks if block.get("status") == "matured_later_block"]
    pre_support = str(pre_june.get("support"))
    if failures:
        verdict = "invalid_evidence_contract"
        active_contract_recommendation = "keep_current_until_evidence_fixed"
    elif pre_support in {"global_rank", "mixed_or_weak_global_rank"} and "timestamp_rank" in later_supports:
        verdict = "conflicting_evidence_keep_timestamp_provisional"
        active_contract_recommendation = "short_boll_timestamp_rank"
    elif pre_support == "global_rank" and matured_later and all(block.get("support") == "global_rank" for block in matured_later):
        verdict = "global_rank_candidate_promotable_pending_manual_review"
        active_contract_recommendation = "anchor_global_policy_rank_reference"
    elif pre_support == "timestamp_rank" and all(block.get("support") in {"timestamp_rank", "tie_or_unknown"} for block in later_blocks):
        verdict = "timestamp_rank_supported_but_still_provisional"
        active_contract_recommendation = "short_boll_timestamp_rank"
    else:
        verdict = "insufficient_or_mixed_evidence_keep_timestamp_provisional"
        active_contract_recommendation = "short_boll_timestamp_rank"

    promotion_gate_passed = verdict == "global_rank_candidate_promotable_pending_manual_review"
    payload = {
        "generated_by": "audit_t1_rank_contract_evidence",
        "pre_june_walkforward": pre_june,
        "later_blocks": later_blocks,
        "thresholds": {
            "min_prejune_folds": thresholds.min_prejune_folds,
            "min_later_timestamps": thresholds.min_later_timestamps,
            "min_later_base_trades": thresholds.min_later_base_trades,
            "min_later_challenger_trades": thresholds.min_later_challenger_trades,
            "min_positive_fold_share": thresholds.min_positive_fold_share,
        },
        "verdict": verdict,
        "promotion_gate_passed": promotion_gate_passed,
        "active_contract_recommendation": active_contract_recommendation,
        "global_rank_promoted": promotion_gate_passed,
        "timestamp_rank_remains_provisional": active_contract_recommendation == "short_boll_timestamp_rank",
        "failures": failures,
    }
    return payload, failures


def _write_report(payload: dict[str, Any], output_dir: Path) -> str:
    pre = payload["pre_june_walkforward"]
    verdict = str(payload.get("verdict", ""))
    interpretation_by_verdict = {
        "invalid_evidence_contract": (
            "At least one evidence contract failed. Do not use this audit for "
            "rank-contract promotion until the listed failures are fixed."
        ),
        "conflicting_evidence_keep_timestamp_provisional": (
            "Global rank won the existing pre-June walk-forward validation, but "
            "later fixed-contract blocks currently support the timestamp-rank "
            "T1 replay. This is conflicting evidence, so the active timestamp "
            "rank should remain provisional and the global rank should stay a "
            "validation challenger."
        ),
        "global_rank_candidate_promotable_pending_manual_review": (
            "Pre-June and sufficiently matured later fixed-contract evidence "
            "both support the global-over-time rank challenger. This makes "
            "global rank a promotion candidate pending manual review and the "
            "remaining production-governance checks."
        ),
        "timestamp_rank_supported_but_still_provisional": (
            "The available evidence supports the timestamp-rank T1 contract, "
            "but the contract remains provisional until additional later "
            "matured windows confirm it."
        ),
        "insufficient_or_mixed_evidence_keep_timestamp_provisional": (
            "The evidence is insufficient or mixed. Keep the active timestamp "
            "rank provisional and continue collecting fixed-contract later "
            "blocks."
        ),
    }
    interpretation = interpretation_by_verdict.get(
        verdict,
        "The evidence does not map to a known promotion verdict. Keep the active rank contract unchanged.",
    )
    lines = [
        "# T1 Rank-Contract Evidence Audit",
        "",
        "This audit combines pre-June walk-forward rank-contract validation with later fixed-contract comparison blocks. It does not rerun replay and does not switch production.",
        "",
        "## Verdict",
        "",
        f"- Verdict: `{payload['verdict']}`",
        f"- Promotion gate passed: `{payload['promotion_gate_passed']}`",
        f"- Active recommendation: `{payload['active_contract_recommendation']}`",
        f"- Failures: `{len(payload['failures'])}`",
        "",
        "## Pre-June Walk-Forward",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| support | {pre['support']} |",
        f"| folds | {pre['fold_count']} |",
        f"| timestamp total net PnL | {pre['timestamp_total_net_pnl']:.6f} |",
        f"| global total net PnL | {pre['global_total_net_pnl']:.6f} |",
        f"| global - timestamp net PnL | {pre['global_minus_timestamp_net_pnl']:.6f} |",
        f"| median fold delta | {pre['median_delta_net_pnl']:.6f} |",
        f"| q25 fold delta | {pre['q25_delta_net_pnl']:.6f} |",
        f"| positive fold share | {pre['positive_delta_share']:.6f} |",
        f"| mean accepted Jaccard | {pre['mean_accepted_jaccard']:.6f} |",
        "",
        "## Later Blocks",
        "",
        "| block | status | timestamps | candidate rows | base trades | challenger trades | support | global - timestamp net PnL | accepted Jaccard |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    for block in payload["later_blocks"]:
        candidate_rows = block.get("candidate_universe_rows")
        candidate_rows_text = "" if candidate_rows is None else str(int(candidate_rows))
        lines.append(
            f"| {Path(block['comparison_dir']).name} | {block['status']} | "
            f"{int(block['timestamp_count'])} | {candidate_rows_text} | {int(block['base_accepted'])} | "
            f"{int(block['challenger_accepted'])} | {block['support']} | "
            f"{float(block['global_minus_timestamp_net_pnl']):.6f} | "
            f"{float(block['accepted_jaccard']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            interpretation,
            "",
            "Generated files:",
            f"- `{output_dir / 't1_rank_contract_evidence_audit.json'}`",
            f"- `{output_dir / 't1_rank_contract_evidence_later_blocks.csv'}`",
            f"- `{output_dir / 't1_rank_contract_evidence_audit.md'}`",
        ]
    )
    if payload["failures"]:
        lines.extend(["", "## Failures", ""])
        for failure in payload["failures"]:
            lines.append(f"- {failure}")
    return "\n".join(lines) + "\n"


def write_rank_contract_evidence(payload: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output_dir / "t1_rank_contract_evidence_audit.json",
        "later_blocks": output_dir / "t1_rank_contract_evidence_later_blocks.csv",
        "report": output_dir / "t1_rank_contract_evidence_audit.md",
    }
    paths["json"].write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(payload["later_blocks"]).to_csv(paths["later_blocks"], index=False)
    paths["report"].write_text(_write_report(payload, output_dir), encoding="utf-8")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-june-walkforward-dir", type=Path, required=True)
    parser.add_argument("--later-comparison-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-prejune-folds", type=int, default=3)
    parser.add_argument("--min-later-timestamps", type=int, default=24)
    parser.add_argument("--min-later-base-trades", type=int, default=30)
    parser.add_argument("--min-later-challenger-trades", type=int, default=30)
    parser.add_argument("--min-positive-fold-share", type=float, default=0.60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    thresholds = EvidenceThresholds(
        min_prejune_folds=int(args.min_prejune_folds),
        min_later_timestamps=int(args.min_later_timestamps),
        min_later_base_trades=int(args.min_later_base_trades),
        min_later_challenger_trades=int(args.min_later_challenger_trades),
        min_positive_fold_share=float(args.min_positive_fold_share),
    )
    payload, failures = build_rank_contract_evidence(
        pre_june_walkforward_dir=args.pre_june_walkforward_dir,
        later_comparison_dirs=list(args.later_comparison_dir),
        thresholds=thresholds,
    )
    paths = write_rank_contract_evidence(payload, args.output_dir)
    print(f"Wrote T1 rank-contract evidence audit: {paths['report']}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
