from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.package_s52_pnl_override_promotion import build_package


def test_pnl_override_package_selects_default_and_risk_actions(tmp_path: Path) -> None:
    handoff_root = tmp_path / "handoff"
    readiness_dir = tmp_path / "readiness"
    replay_dir = tmp_path / "replay"
    handoff_root.mkdir()
    readiness_dir.mkdir()
    replay_dir.mkdir()

    pd.DataFrame(
        {
            "variant": ["top5", "top10"],
            "gate3_status": ["pnl_override_candidate", "pnl_override_candidate"],
            "path_risk_status": ["bad_mae_accepted_by_pnl_override", "bad_mae_accepted_by_pnl_override"],
            "bad_mae_accepted_by_pnl_override": [True, True],
            "failed_checks": ["month_bad_mae_bar", "month_bad_mae_bar"],
        }
    ).to_csv(readiness_dir / "s52_meta_handoff_gate3_readiness_summary.csv", index=False)
    pd.DataFrame(
        {
            "variant": ["top5", "top10"],
            "rows": [100, 220],
            "matched_rows": [100, 220],
            "unmatched_rows": [0, 0],
            "symbols": [50, 90],
            "sum_net_pnl": [2.0, 3.2],
            "mean_ret_net": [0.020, 0.016],
            "worst_month_ret_net": [0.014, 0.010],
            "mean_exec_margin": [0.016, 0.012],
            "worst_month_exec_margin": [0.010, 0.006],
            "hit_rate_ret_net": [0.80, 0.77],
            "positive_exec_margin_rate": [0.72, 0.69],
            "full_path_bad_mae": [0.45, 0.56],
            "max_month_full_path_bad_mae": [0.54, 0.58],
            "timeout": [0.01, 0.01],
            "dominant_side_share": [0.79, 0.79],
            "clean_handoff_has_no_realized_outcomes": [True, True],
            "offline_parity_key_set_match": [True, True],
        }
    ).to_csv(replay_dir / "s52_frozen_action_replay_summary.csv", index=False)
    pd.DataFrame(
        {
            "variant": ["top10", "top10", "top10", "top10"],
            "grouping": ["side_source", "side_aegmm", "side_reconstruction", "side_leaf_exec_margin"],
            "side_name": ["short", "short", "long", "short"],
            "source_semantic_family": ["mean_reversion", None, None, None],
            "aegmm_cluster": [None, "c1", None, None],
            "reconstruction_bin": [None, None, "q3", None],
            "regime_lgbm_leaf_exec_margin_k4": [None, None, None, "c9"],
            "rows": [120, 60, 40, 5],
            "symbols": [50, 30, 25, 5],
            "mean_ret_net": [0.018, 0.012, -0.002, 0.050],
            "worst_month_ret_net": [0.010, 0.008, -0.004, 0.050],
            "mean_exec_margin": [0.014, 0.008, -0.006, 0.046],
            "hit_rate_ret_net": [0.80, 0.70, 0.40, 1.0],
            "positive_exec_margin_rate": [0.72, 0.62, 0.30, 1.0],
            "full_path_bad_mae": [0.48, 0.63, 0.70, 0.0],
            "timeout": [0.01, 0.02, 0.01, 0.0],
            "inferred_decision_stop_touch": [0.20, 0.30, 0.50, 0.0],
            "inferred_full_path_stop_touch": [0.48, 0.65, 0.75, 0.0],
        }
    ).to_csv(replay_dir / "s52_frozen_action_replay_breakdown.csv", index=False)
    pd.DataFrame(
        {
            "regime_model": ["aegmm_cluster"],
            "recommended_action": ["upweight_or_lower_meta_threshold_candidate"],
            "promotion_status": ["meta_context_candidate"],
            "validation_status": ["holdout_confirms_margin"],
            "source_tag": ["short__mean_reversion"],
            "fit_rows": [100],
            "holdout_rows": [20],
            "expected_delta_exec_margin": [0.002],
            "expected_delta_full_path_bad_mae": [-0.03],
        }
    ).to_csv(handoff_root / "policy_recommendation_table.csv", index=False)

    manifest = build_package(
        handoff_root=handoff_root,
        readiness_dir=readiness_dir,
        replay_dir=replay_dir,
        out_dir=tmp_path / "out",
    )
    candidates = pd.read_csv(manifest["outputs"]["candidate_decision"])
    actions = pd.read_csv(manifest["outputs"]["source_regime_actions"])

    roles = dict(zip(candidates["variant"], candidates["promotion_role"]))
    assert roles["top10"] == "default_capacity_candidate"
    assert roles["top5"] == "conservative_precision_benchmark"
    assert set(candidates["promotion_status"]) == {"promote_to_meta_execution_integration_candidate"}
    assert not actions["hard_gate_allowed"].astype(bool).any()
    by_group = dict(zip(actions["grouping"], actions["recommended_action"]))
    assert by_group["side_source"] == "feature_plus_normal_size"
    assert by_group["side_aegmm"] == "feature_plus_strong_size_down"
    assert by_group["side_reconstruction"] == "feature_plus_strong_downweight"
    assert by_group["side_leaf_exec_margin"] == "feature_only_low_support"
    assert manifest["decision"]["hard_gates_allowed"] is False
