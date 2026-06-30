import json
from pathlib import Path

import pandas as pd

from scripts.audit_t1_rank_contract_evidence import (
    EvidenceThresholds,
    build_rank_contract_evidence,
    write_rank_contract_evidence,
)


def _write_prejune(root: Path, *, global_better: bool = True, unsafe_leakage: bool = False) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generated_by": "run_t1_rank_contract_walkforward",
        "purpose": "pre_june_rank_contract_validation",
        "fold_count": 3,
        "arms": {
            "timestamp_rank_t1": {
                "rank_contract": "short_boll_timestamp_rank",
                "rank_scope": "within_timestamp",
            },
            "fold_causal_global_rank_reference": {
                "rank_contract": "fold_causal_global_rank_reference",
                "rank_scope": "global_over_time",
                "fit_scope": "training_timestamps_only_per_fold",
            },
        },
        "fixed_policy_contract": {
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
        },
        "leakage_contract": {
            "split_by_complete_timestamps": True,
            "global_rank_reference_uses_validation_rows": unsafe_leakage,
            "global_rank_reference_uses_future_rows": False,
            "market_state_controller_active": False,
            "qfail_active": False,
            "rank_contract_is_the_only_arm_difference": True,
        },
    }
    (root / "t1_rank_contract_walkforward_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    timestamp_pnl = 100.0
    global_pnl = 130.0 if global_better else 80.0
    pd.DataFrame(
        {
            "arm": ["timestamp_rank_t1", "fold_causal_global_rank_reference"],
            "folds": [3, 3],
            "total_trades": [30, 30],
            "total_net_pnl": [timestamp_pnl, global_pnl],
            "median_fold_net_pnl": [30.0, 40.0],
            "q25_fold_net_pnl": [20.0, 30.0],
            "positive_fold_share": [1.0, 1.0],
        }
    ).to_csv(root / "rank_contract_walkforward_aggregate.csv", index=False)
    deltas = [10.0, 8.0, 12.0] if global_better else [-4.0, -5.0, -6.0]
    pd.DataFrame(
        {
            "fold": [1, 2, 3],
            "delta_net_pnl": deltas,
            "accepted_jaccard": [0.5, 0.6, 0.7],
        }
    ).to_csv(root / "rank_contract_walkforward_fold_delta.csv", index=False)
    pd.DataFrame(
        {
            "fold": [1, 2, 3],
            "jaccard": [0.5, 0.6, 0.7],
            "removed_net_pnl": [1.0, 2.0, 3.0],
            "added_net_pnl": [4.0, 5.0, 6.0],
        }
    ).to_csv(root / "rank_contract_walkforward_accepted_overlap.csv", index=False)
    pd.DataFrame(
        {
            "fold": [1, 2, 3],
            "global_valid_missing_policy_rank_rows": [0, 0, 0],
            "global_valid_missing_auction_rank_rows": [0, 0, 0],
            "global_train_missing_policy_rank_rows": [0, 0, 0],
            "global_train_missing_auction_rank_rows": [0, 0, 0],
            "global_valid_ranked_rows": [10, 10, 10],
            "valid_broad_rows": [10, 10, 10],
            "global_train_ranked_rows": [20, 20, 20],
            "train_deployable_rows": [20, 20, 20],
        }
    ).to_csv(root / "rank_contract_walkforward_rank_diagnostics.csv", index=False)
    return root


def _write_later(
    root: Path,
    *,
    global_delta: float,
    timestamps: int = 9,
    trades: int = 8,
    unsafe_contract: bool = False,
    unsafe_rank_reference: bool = False,
    unsafe_candidate_universe: bool = False,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    fixed_contract = {
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
    if unsafe_contract:
        fixed_contract["auction"] = "head_priority_auction"
    rank_reference_contract = {
        "required": True,
        "passed": not unsafe_rank_reference,
        "failures": [] if not unsafe_rank_reference else ["eval.window_rank_debug_used"],
        "eval_expected_rows": 20,
        "train_deployable_expected_rows": 40,
        "eval_rank_reference_run_id": "prejune_ref",
        "train_rank_reference_run_id": "prejune_ref",
        "eval_ranked_rows": 20,
        "train_ranked_rows": 40,
        "eval_auction_ranked_rows": 20,
        "train_auction_ranked_rows": 40,
    }
    rank_reference_diagnostics = {
        "eval": {
            "rank_reference_run_id": "prejune_ref",
            "rank_source": "policy_rank_reference_percentile",
            "missing_rank_rows": 0,
            "missing_auction_rank_rows": 0,
            "ranked_rows": 20,
            "auction_ranked_rows": 20,
            "window_rank_debug_used": bool(unsafe_rank_reference),
        },
        "train_deployable": {
            "rank_reference_run_id": "prejune_ref",
            "rank_source": "policy_rank_reference_percentile",
            "missing_rank_rows": 0,
            "missing_auction_rank_rows": 0,
            "ranked_rows": 40,
            "auction_ranked_rows": 40,
            "window_rank_debug_used": False,
        },
    }
    (root / "rank_contract_comparison_manifest.json").write_text(
        json.dumps(
            {
                "generated_by": "compare_t1_rank_contracts",
                "purpose": "later_fixed_contract_timestamp_vs_global_rank_validation",
                "base": {
                    "rank_contract": "short_boll_timestamp_rank",
                    "rank_scope": "within_timestamp",
                },
                "challenger": {
                    "rank_contract": "anchor_global_policy_rank_reference",
                    "rank_scope": "global_over_time",
                    "rank_reference_contract": rank_reference_contract,
                    "rank_reference_diagnostics": rank_reference_diagnostics,
                },
                "fixed_policy_contract": fixed_contract,
                "candidate_universe": {
                    "overlap": {
                        "base_keys": 20,
                        "challenger_keys": 21 if unsafe_candidate_universe else 20,
                        "intersection_keys": 20,
                        "union_keys": 21 if unsafe_candidate_universe else 20,
                        "base_only_keys": 0,
                        "challenger_only_keys": 1 if unsafe_candidate_universe else 0,
                        "jaccard": 20 / 21 if unsafe_candidate_universe else 1.0,
                        "identical": not unsafe_candidate_universe,
                    }
                },
                "validation": {
                    "passed": not (unsafe_contract or unsafe_rank_reference or unsafe_candidate_universe),
                    "failures": (
                        []
                        if not (unsafe_contract or unsafe_rank_reference or unsafe_candidate_universe)
                        else [
                            *(
                                ["fixed_policy_contract_mismatch_auction"]
                                if unsafe_contract
                                else []
                            ),
                            *(["challenger.eval_window_rank_debug_used"] if unsafe_rank_reference else []),
                            *(["candidate_universe_not_identical"] if unsafe_candidate_universe else []),
                        ]
                    ),
                    "rank_contract_is_the_only_arm_difference": not (
                        unsafe_contract or unsafe_rank_reference or unsafe_candidate_universe
                    ),
                    "score_threshold_auction_ev_costs_fixed": not unsafe_contract,
                    "candidate_universe_identical": not unsafe_candidate_universe,
                    "controller_and_qfail_disabled": True,
                },
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "contract_name": ["timestamp", "global"],
            "trade_count": [trades, trades],
            "net_pnl": [100.0, 100.0 + global_delta],
            "gross_pnl": [120.0, 120.0 + global_delta],
            "cost_pnl": [20.0, 20.0],
            "full_sl_rate": [0.1, 0.1],
            "timeout_rate": [0.1, 0.1],
            "worst_24h_net_pnl": [-5.0, -5.0],
        }
    ).to_csv(root / "rank_contract_summary.csv", index=False)
    pd.DataFrame(
        {
            "metric": ["trade_count", "net_pnl", "full_sl_rate"],
            "challenger_minus_base": [0.0, global_delta, 0.0],
        }
    ).to_csv(root / "rank_contract_delta.csv", index=False)
    (root / "rank_contract_accepted_overlap.json").write_text(
        json.dumps(
            {
                "overlap": {
                    "base_accepted": trades,
                    "challenger_accepted": trades,
                    "jaccard": 0.5,
                    "base_only": 2,
                    "challenger_only": 2,
                },
                "swap_pnl": {
                    "removed_net_pnl": 1.0,
                    "added_net_pnl": 1.0 + global_delta,
                    "removed_winner_pnl": 2.0,
                    "added_loser_loss": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "contract_name": ["timestamp", "timestamp", "global", "global"],
            "head": ["short_asset", "short_boll", "short_asset", "short_boll"],
            "deployable_rows": [10, 10, 10, 10],
            "timestamp_count": [timestamps, timestamps, timestamps, timestamps],
        }
    ).to_csv(root / "rank_contract_deployable_rows.csv", index=False)
    return root


def test_rank_contract_evidence_keeps_timestamp_provisional_on_conflicting_evidence(tmp_path: Path) -> None:
    prejune = _write_prejune(tmp_path / "prejune", global_better=True)
    later = _write_later(tmp_path / "later", global_delta=-10.0, timestamps=9, trades=8)

    payload, failures = build_rank_contract_evidence(
        pre_june_walkforward_dir=prejune,
        later_comparison_dirs=[later],
        thresholds=EvidenceThresholds(),
    )

    assert failures == []
    assert payload["verdict"] == "conflicting_evidence_keep_timestamp_provisional"
    assert payload["promotion_gate_passed"] is False
    assert payload["active_contract_recommendation"] == "short_boll_timestamp_rank"
    assert payload["later_blocks"][0]["status"] == "informative_small_sample_not_promotion"


def test_rank_contract_evidence_can_mark_global_candidate_when_all_matured_evidence_agrees(tmp_path: Path) -> None:
    prejune = _write_prejune(tmp_path / "prejune", global_better=True)
    later = _write_later(tmp_path / "later", global_delta=10.0, timestamps=48, trades=40)

    payload, failures = build_rank_contract_evidence(
        pre_june_walkforward_dir=prejune,
        later_comparison_dirs=[later],
        thresholds=EvidenceThresholds(),
    )

    assert failures == []
    assert payload["verdict"] == "global_rank_candidate_promotable_pending_manual_review"
    assert payload["promotion_gate_passed"] is True
    assert payload["active_contract_recommendation"] == "anchor_global_policy_rank_reference"


def test_rank_contract_evidence_rejects_leaky_prejune_reference(tmp_path: Path) -> None:
    prejune = _write_prejune(tmp_path / "prejune", global_better=True, unsafe_leakage=True)
    later = _write_later(tmp_path / "later", global_delta=10.0, timestamps=48, trades=40)

    payload, failures = build_rank_contract_evidence(
        pre_june_walkforward_dir=prejune,
        later_comparison_dirs=[later],
        thresholds=EvidenceThresholds(),
    )

    assert "pre-June leakage_contract.global_rank_reference_uses_validation_rows != False" in failures
    assert payload["verdict"] == "invalid_evidence_contract"
    assert payload["promotion_gate_passed"] is False


def test_rank_contract_evidence_rejects_later_comparison_with_changed_policy_contract(tmp_path: Path) -> None:
    prejune = _write_prejune(tmp_path / "prejune", global_better=True)
    later = _write_later(tmp_path / "later", global_delta=10.0, timestamps=48, trades=40, unsafe_contract=True)

    payload, failures = build_rank_contract_evidence(
        pre_june_walkforward_dir=prejune,
        later_comparison_dirs=[later],
        thresholds=EvidenceThresholds(),
    )

    assert "later comparison manifest validation.passed is not true" in failures
    assert "later fixed_policy_contract.auction != global_auction" in failures
    assert payload["verdict"] == "invalid_evidence_contract"


def test_rank_contract_evidence_rejects_later_global_rank_debug_fallback(tmp_path: Path) -> None:
    prejune = _write_prejune(tmp_path / "prejune", global_better=True)
    later = _write_later(
        tmp_path / "later",
        global_delta=10.0,
        timestamps=48,
        trades=40,
        unsafe_rank_reference=True,
    )

    payload, failures = build_rank_contract_evidence(
        pre_june_walkforward_dir=prejune,
        later_comparison_dirs=[later],
        thresholds=EvidenceThresholds(),
    )

    assert "later comparison manifest validation.passed is not true" in failures
    assert "later comparison challenger rank_reference_contract.passed is not true" in failures
    assert "later comparison challenger eval used window-rank debug fallback" in failures
    assert payload["verdict"] == "invalid_evidence_contract"


def test_rank_contract_evidence_rejects_later_candidate_universe_mismatch(tmp_path: Path) -> None:
    prejune = _write_prejune(tmp_path / "prejune", global_better=True)
    later = _write_later(
        tmp_path / "later",
        global_delta=10.0,
        timestamps=48,
        trades=40,
        unsafe_candidate_universe=True,
    )

    payload, failures = build_rank_contract_evidence(
        pre_june_walkforward_dir=prejune,
        later_comparison_dirs=[later],
        thresholds=EvidenceThresholds(),
    )

    assert "later comparison manifest validation.passed is not true" in failures
    assert "later comparison manifest candidate_universe_identical is not true" in failures
    assert payload["verdict"] == "invalid_evidence_contract"


def test_rank_contract_evidence_report_interpretation_follows_verdict(tmp_path: Path) -> None:
    prejune = _write_prejune(tmp_path / "prejune", global_better=True)
    later = _write_later(tmp_path / "later", global_delta=10.0, timestamps=48, trades=40)
    payload, failures = build_rank_contract_evidence(
        pre_june_walkforward_dir=prejune,
        later_comparison_dirs=[later],
        thresholds=EvidenceThresholds(),
    )

    assert failures == []
    paths = write_rank_contract_evidence(payload, tmp_path / "audit")
    report = paths["report"].read_text(encoding="utf-8")

    assert "global rank a promotion candidate" in report
    assert "conflicting evidence" not in report
