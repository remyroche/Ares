#!/usr/bin/env python3
"""Exact-policy portfolio sweep for grouped-OOF model-failure risk.

This runner is deliberately separate from the active-transition policy runner:
economic model failure and market transition are distinct targets.  It uses
the canonical February-April 2025 raw-alpha/exact-policy lineage, one pooled
global top-k and the shared portfolio constraints.  Grouped OOF failure scores
support research only, never promotion.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ARES_ROOT = Path(
    os.environ.get("ARES_ROOT", "/Users/remyroche/Documents/Ares")
).resolve()
if str(ARES_ROOT) not in sys.path:
    sys.path.insert(0, str(ARES_ROOT))

from extreme_price_movements.portfolio_policy_replay import (
    load_portfolio_policy_params,
    replay_candidates,
)
from run_active_transition_canonical_policy_sweep import (
    IDENTITY_EV_CURVE,
    _accepted_with_metadata,
    _book_metrics,
    _conditional_accepted_metrics,
    _floats,
    _monthly_accepted_metrics,
    _robust_score_scale,
    _safe,
    _sha256,
    _stable_top_k,
    _write_json,
    replacement_attribution,
    select_arm,
    to_replay_candidates,
)


def attach_failure_risk(
    candidates: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    failure_label: str,
    feature_set: str,
) -> pd.DataFrame:
    probability_column = f"prediction__{failure_label}__{feature_set}"
    target_column = f"target__economic_failure_{failure_label}_active"
    event_column = f"target__economic_failure_{failure_label}_event_id"
    required = {
        "source_utc",
        probability_column,
        target_column,
        event_column,
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"failure predictions lack {missing}")
    local = predictions[
        ["source_utc", probability_column, target_column, event_column]
    ].copy()
    local["source_utc"] = pd.to_datetime(
        local["source_utc"], utc=True, errors="raise"
    )
    if local["source_utc"].duplicated().any():
        raise ValueError("failure prediction must have one row per source hour")
    local = local.rename(
        columns={
            probability_column: "failure_probability_oof",
            target_column: "expost_failure_active",
            event_column: "failure_event_id",
        }
    )
    work = candidates.copy()
    work["__ts__"] = pd.to_datetime(work["__ts__"], utc=True, errors="raise")
    work = work.merge(
        local,
        left_on="__ts__",
        right_on="source_utc",
        how="inner",
        validate="many_to_one",
    )
    if work["failure_probability_oof"].isna().any():
        raise ValueError("failure risk join contains missing probabilities")
    # Internal aliases allow reuse of the exact, tested selection/replay
    # mechanics.  All persisted user-facing ledgers are renamed below.
    work["active_transition_probability_oof"] = work[
        "failure_probability_oof"
    ]
    work["expost_transition_active"] = work[
        "expost_failure_active"
    ].astype(np.int8)
    work["transition_event_id"] = work["failure_event_id"]
    return work


def _rename_failure_semantics(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(
        columns={
            "active_transition_probability_oof": "failure_probability_oof_alias",
            "expost_transition_active": "expost_failure_active_alias",
            "transition_event_id": "failure_event_id_alias",
        }
    )


def _failure_metric_keys(metrics: dict[str, Any]) -> dict[str, Any]:
    """Remove transition-only terminology from persisted failure reports."""
    return {
        key.replace("active_", "failure_"): value
        for key, value in metrics.items()
    }


def _condition_rows(
    accepted: pd.DataFrame,
    *,
    score_stream: str,
    arm: str,
    policy: str,
    value: float,
) -> list[dict[str, Any]]:
    rows = _conditional_accepted_metrics(accepted)
    mapping = {
        "true_active_transition": "true_economic_failure",
        "outside_true_transition": "outside_economic_failure",
        "predicted_active_ge_0p5": "predicted_failure_ge_0p5",
    }
    return [
        {
            "score_stream": score_stream,
            "arm": arm,
            "policy": policy,
            "value": value,
            **{**row, "condition": mapping.get(row["condition"], row["condition"])},
        }
        for row in rows
    ]


def _monthly_rows(
    accepted: pd.DataFrame,
    *,
    score_stream: str,
    arm: str,
    policy: str,
    value: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _monthly_accepted_metrics(accepted):
        renamed = {
            key.replace("active_", "failure_"): item
            for key, item in row.items()
        }
        rows.append(
            {
                "score_stream": score_stream,
                "arm": arm,
                "policy": policy,
                "value": value,
                **renamed,
            }
        )
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    candidates_path = Path(args.candidates)
    predictions_path = Path(args.failure_predictions)
    portfolio_path = Path(args.portfolio_config)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    cohort = attach_failure_risk(
        pd.read_parquet(candidates_path),
        pd.read_parquet(predictions_path),
        failure_label=args.failure_label,
        feature_set=args.feature_set,
    )
    cohort = cohort.loc[cohort["mapped_eligible"].astype(bool)].copy()
    if args.evaluation_start:
        cohort = cohort.loc[
            cohort["__ts__"].ge(pd.Timestamp(args.evaluation_start, tz="UTC"))
        ].copy()
    if args.evaluation_end:
        cohort = cohort.loc[
            cohort["__ts__"].lt(pd.Timestamp(args.evaluation_end, tz="UTC"))
        ].copy()
    if cohort.empty:
        raise ValueError("evaluation window contains no candidates")
    failure_events = int(
        cohort.loc[
            cohort["expost_failure_active"].eq(1), "failure_event_id"
        ].dropna().nunique()
    )
    if failure_events < int(args.minimum_failure_events):
        raise ValueError(
            f"only {failure_events} failure events overlap evaluation"
        )
    params = replace(
        load_portfolio_policy_params(portfolio_path),
        enforce_position_count_cap=True,
    )
    specifications = [("baseline", 0.0)]
    if args.policy_selection_contract == "prior_frozen":
        if args.frozen_policy is None or args.frozen_value is None:
            raise ValueError("prior_frozen requires policy and value")
        specifications.append((args.frozen_policy, float(args.frozen_value)))
    else:
        lambdas = _floats(args.lambdas)
        for policy in (
            "trust_discount",
            "risk_premium",
            "threshold_increase",
            "exposure_reduction",
        ):
            specifications.extend((policy, value) for value in lambdas)
    output.mkdir(parents=True, exist_ok=False)
    baseline = _stable_top_k(
        cohort,
        score_column=args.score_column,
        fraction=float(args.top_k_fraction),
    )
    baseline_ids = set(baseline["candidate_id"].astype(str))
    baseline_count = int(len(baseline))
    score_scale = _robust_score_scale(cohort, args.score_column)
    summary_rows: list[dict[str, Any]] = []
    conditional_rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    accepted_by_arm: dict[str, pd.DataFrame] = {}
    for policy, value in specifications:
        arm = f"{policy}_{value:.4f}".replace(".", "p")
        selected = select_arm(
            cohort,
            score_column=args.score_column,
            baseline_ids=baseline_ids,
            baseline_count=baseline_count,
            score_scale=score_scale,
            policy=policy,
            value=value,
        )
        selected_ids = set(selected["candidate_id"].astype(str))
        replacement_metrics, replacement = replacement_attribution(
            cohort,
            baseline_ids=baseline_ids,
            selected_ids=selected_ids,
        )
        relation = replacement.set_index("candidate_id")[
            "selection_relation"
        ]
        selected["selection_relation"] = selected["candidate_id"].map(relation)
        replay_frame = to_replay_candidates(selected)
        decisions, equity, metrics = replay_candidates(
            replay_frame,
            params,
            mode="global_auction",
            ev_curve=IDENTITY_EV_CURVE,
            initial_wallet=float(args.initial_wallet),
            market_mode="perps",
        )
        accepted = _accepted_with_metadata(decisions, replay_frame)
        accepted_by_arm[arm] = accepted
        persisted_selected = _rename_failure_semantics(selected)
        persisted_accepted = _rename_failure_semantics(accepted)
        persisted_replacement = _rename_failure_semantics(replacement)
        persisted_selected.to_parquet(
            output / f"{arm}_selected.parquet", index=False
        )
        persisted_accepted.to_parquet(
            output / f"{arm}_accepted.parquet", index=False
        )
        persisted_replacement.to_parquet(
            output / f"{arm}_replacement_attribution.parquet", index=False
        )
        equity.to_parquet(output / f"{arm}_equity.parquet", index=False)
        summary_rows.append(
            {
                "score_stream": args.score_column,
                "failure_label": args.failure_label,
                "failure_feature_set": args.feature_set,
                "score_scale": score_scale,
                "arm": arm,
                "policy": policy,
                "value": float(value),
                **_failure_metric_keys(
                    _book_metrics(selected, prefix="selected")
                ),
                **_failure_metric_keys(replacement_metrics),
                **{
                    key: metric
                    for key, metric in metrics.items()
                    if isinstance(
                        metric, (str, int, float, bool, np.generic)
                    )
                },
            }
        )
        conditional_rows.extend(
            _condition_rows(
                accepted,
                score_stream=args.score_column,
                arm=arm,
                policy=policy,
                value=float(value),
            )
        )
        monthly_rows.extend(
            _monthly_rows(
                accepted,
                score_stream=args.score_column,
                arm=arm,
                policy=policy,
                value=float(value),
            )
        )
    summary = pd.DataFrame(summary_rows)
    baseline_row = summary.loc[summary["policy"].eq("baseline")].iloc[0]
    baseline_accepted = accepted_by_arm["baseline_0p0000"].set_index(
        "candidate_id"
    )
    for index, row in summary.iterrows():
        arm = row["arm"]
        accepted = accepted_by_arm[arm]
        missed = set(baseline_accepted.index.astype(str)).difference(
            accepted["candidate_id"].astype(str)
        )
        missed_return = pd.to_numeric(
            baseline_accepted.reindex(list(missed))["position_net_return"],
            errors="coerce",
        )
        summary.loc[index, "missed_baseline_accepted_trades"] = len(missed)
        summary.loc[index, "missed_profitable_trades"] = int(
            missed_return.gt(0.0).sum()
        )
        summary.loc[index, "missed_profitable_return_sum"] = float(
            missed_return.loc[missed_return.gt(0.0)].sum()
        )
        for metric in (
            "net_pnl",
            "compounded_return",
            "sortino",
            "max_drawdown",
            "worst_week",
            "notional_turnover",
            "trade_count",
        ):
            summary.loc[index, f"delta_{metric}"] = (
                float(row[metric]) - float(baseline_row[metric])
            )
    summary_path = output / "policy_summary.csv"
    conditional_path = output / "conditional_economics.csv"
    monthly_path = output / "monthly_economics.csv"
    summary.to_csv(summary_path, index=False)
    pd.DataFrame(conditional_rows).to_csv(conditional_path, index=False)
    pd.DataFrame(monthly_rows).to_csv(monthly_path, index=False)
    policy_blocker = (
        "lambda grid evaluated on development cohort"
        if args.policy_selection_contract == "same_cohort_grid"
        else "policy/lambda frozen on declared prior cohort"
    )
    manifest = {
        "schema": "model_failure_risk_exact_policy_sweep_v1",
        "status": "RESEARCH_ONLY_GROUPED_FAILURE_OOF_POLICY_COMPLETE",
        "promotion_eligible": False,
        "promotion_blocker": (
            "failure risk is grouped OOF on February-April raw-alpha lineage, "
            f"not chronological current-lineage OOS; {policy_blocker}"
        ),
        "risk_contract": {
            "failure_label": args.failure_label,
            "feature_set": args.feature_set,
            "validation": "whole failure windows grouped OOF",
        },
        "selection_contract": (
            "one pooled global top-k across both sides and all timestamps"
        ),
        "evaluation_window": {
            "start_inclusive": args.evaluation_start,
            "end_exclusive": args.evaluation_end,
            "failure_events": failure_events,
        },
        "policy_selection_contract": {
            "name": args.policy_selection_contract,
            "frozen_policy": args.frozen_policy,
            "frozen_value": args.frozen_value,
        },
        "portfolio_contract": {
            "configuration": str(portfolio_path),
            "max_concurrent_positions": params.max_concurrent_positions,
            "max_concurrent_per_symbol": params.max_concurrent_per_symbol,
            "max_new_entries_per_bar": params.max_new_entries_per_bar,
            "max_total_wallet_allocation_pct": (
                params.max_total_wallet_allocation_pct
            ),
        },
        "sources": {
            "candidates": {
                "path": str(candidates_path),
                "sha256": _sha256(candidates_path),
            },
            "failure_predictions": {
                "path": str(predictions_path),
                "sha256": _sha256(predictions_path),
            },
            "portfolio": {
                "path": str(portfolio_path),
                "sha256": _sha256(portfolio_path),
            },
        },
        "outputs": {
            "policy_summary": {
                "path": str(summary_path),
                "sha256": _sha256(summary_path),
            },
            "conditional_economics": {
                "path": str(conditional_path),
                "sha256": _sha256(conditional_path),
            },
            "monthly_economics": {
                "path": str(monthly_path),
                "sha256": _sha256(monthly_path),
            },
        },
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
    }
    manifest_path = output / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    root = Path("/Users/remyroche/Documents/Ares")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=root
        / (
            "data_perp/artifacts/historical_causal_score_economics_mapping_20260729_v1/"
            "canonical_base__score_base_alpha/causal_mapped_candidates.parquet"
        ),
    )
    parser.add_argument(
        "--failure-predictions",
        type=Path,
        default=root
        / (
            "data_perp/artifacts/historical_exact_model_failure_ablation_20260729_v3/"
            "grouped_oof_predictions.parquet"
        ),
    )
    parser.add_argument(
        "--portfolio-config",
        type=Path,
        default=root
        / (
            "data_perp/artifacts/"
            "s59_s52_finalfit_meta_repairedcoverage_v9tail95_mlp_hierev_20260715_v3/"
            "policy_params/optimized_portfolio_policy_config.json"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--failure-label", choices=("broad", "strict"), required=True)
    parser.add_argument("--feature-set", default="market_plus_health")
    parser.add_argument("--score-column", default="score_raw")
    parser.add_argument("--top-k-fraction", type=float, default=0.10)
    parser.add_argument("--lambdas", default="0.25,0.50,1.00")
    parser.add_argument(
        "--policy-selection-contract",
        choices=("same_cohort_grid", "prior_frozen"),
        default="same_cohort_grid",
    )
    parser.add_argument(
        "--frozen-policy",
        choices=(
            "trust_discount",
            "risk_premium",
            "threshold_increase",
            "exposure_reduction",
        ),
    )
    parser.add_argument("--frozen-value", type=float)
    parser.add_argument("--evaluation-start")
    parser.add_argument("--evaluation-end")
    parser.add_argument("--minimum-failure-events", type=int, default=5)
    parser.add_argument("--initial-wallet", type=float, default=10_000.0)
    return parser


def main() -> None:
    print(json.dumps(_safe(run(_parser().parse_args())), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
