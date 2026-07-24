#!/usr/bin/env python3
"""Create a compact, reproducible status table for the frozen challenger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def run(args: argparse.Namespace) -> dict[str, object]:
    args.output.mkdir(parents=True, exist_ok=True)
    frozen = _read_json(args.frozen / "manifest.json")
    credibility = _read_json(args.credibility / "summary.json")
    episodes = _read_json(args.episodes / "manifest.json")
    conditional = _read_json(args.conditional / "manifest.json")
    monitor = _read_json(args.monitor / "manifest.json")
    replication = frozen["replication"]
    parent = replication["short_default_parent"]
    challenger = replication["short_default_challenger"]
    rows = [
        {
            "step": "frozen_oos_replication_all_selected",
            "status": "research_candidate_only",
            "evidence_unit": "all selected trades, Apr-Jun 2026 OOS",
            "metric_1": "total_ev_delta",
            "value_1": replication["delta_sum_ev"],
            "metric_2": "ev_per_trade_delta_pp",
            "value_2": 100.0 * replication["delta_mean_ev"],
            "metric_3": "activity_retained",
            "value_3": replication["activity_retained"],
        },
        {
            "step": "frozen_oos_replication_short_default",
            "status": "improved",
            "evidence_unit": "selected short-default trades",
            "metric_1": "ev_per_trade_delta_pp",
            "value_1": 100.0 * (challenger["mean_ev"] - parent["mean_ev"]),
            "metric_2": "clean_precision_delta_pp",
            "value_2": 100.0 * (challenger["clean_precision"] - parent["clean_precision"]),
            "metric_3": "worst_month_ev_delta_pp",
            "value_3": 100.0 * (challenger["worst_month_ev"] - parent["worst_month_ev"]),
        },
        {
            "step": "trade_weighted_bootstrap",
            "status": "directionally_positive",
            "evidence_unit": "Bayesian bootstrap",
            "metric_1": "joint_pass_probability",
            "value_1": credibility["joint_contract"]["joint_pass_probability"],
            "metric_2": "leave_event_sign_reversal",
            "value_2": credibility["leave_out_flags"]["total_ev_sign_reversals"],
            "metric_3": "largest_event_uplift_share",
            "value_3": credibility["leave_out_flags"]["max_abs_event_block_influence"],
        },
        {
            "step": "equal_block_attribution",
            "status": "fails_deployment_concentration",
            "evidence_unit": "independent high-uncertainty/adverse blocks",
            "metric_1": "blocks",
            "value_1": episodes["episode_count"],
            "metric_2": "p_block_mu_gt_zero",
            "value_2": episodes["equal_block_posterior"]["p_mu_gt_zero"],
            "metric_3": "mean_future_block_positive_probability",
            "value_3": episodes["equal_block_posterior"]["mean_future_block_positive_probability"],
        },
        {
            "step": "conditional_lookalike_discriminator",
            "status": conditional["promotion_status"],
            "evidence_unit": "train-OOF causal context required",
            "metric_1": "missing_train_context_columns",
            "value_1": ";".join(conditional.get("missing_train_columns", [])),
            "metric_2": "candidate_feature_count",
            "value_2": len(conditional.get("available_candidate_features", [])),
            "metric_3": "oos_substitution",
            "value_3": "prohibited",
        },
        {
            "step": "prospective_block_monitor",
            "status": "ready_observational_only",
            "evidence_unit": "frozen score conjunction + cooling rule",
            "metric_1": "historical_conjunction_blocks",
            "value_1": monitor["block_count"],
            "metric_2": "minimum_new_blocks",
            "value_2": monitor["confirmation_contract"]["minimum_new_independent_blocks"],
            "metric_3": "minimum_improving_blocks",
            "value_3": monitor["confirmation_contract"]["minimum_improving_blocks"],
        },
    ]
    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.output / "challenger_status_metrics.csv", index=False)
    lines = [
        "# Frozen Short-Default Challenger Status",
        "",
        "The challenger remains frozen, research-only, and inactive. The OOS replication improves EV and clean precision, but its uplift is concentrated in the April 1-8 regime block.",
        "",
        metrics.to_markdown(index=False),
        "",
        "## Decision",
        "",
        "- Do not deploy or tune this challenger on the present sample.",
        "- Do not run the conditional discriminator until causal train-OOF context contains the exact observable mechanism inputs.",
        "- Count future evidence only when the frozen conjunction starts after the configured cooling period. Outcomes may evaluate a block after resolution, but never define its start or end.",
    ]
    (args.output / "status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "schema": "short_default_challenger_status_report_v1",
        "frozen_candidate_id": frozen["candidate_id"],
        "status": "frozen_research_challenger_not_live",
        "report_rows": int(len(metrics)),
        "sources": {
            "frozen": str(args.frozen),
            "credibility": str(args.credibility),
            "episodes": str(args.episodes),
            "conditional": str(args.conditional),
            "monitor": str(args.monitor),
        },
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--credibility", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--conditional", type=Path, required=True)
    parser.add_argument("--monitor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
