from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts import consolidate_j4_j5_contextual_meta_freeze as mod


def _write_minimal_artifact(path: Path, head: str, *, promoted: bool = False, j5_rows: int = 0) -> None:
    path.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "head": head,
                "decision": "promote_j4_capacity_pending_fresh_oos"
                if promoted
                else "retain_selected_contextual_feature_arm_j4_not_promoted",
                "selected_contextual_feature_arm": "B_current_plus_model_state",
                "selected_capacity_config": "cfg" if promoted else "",
                "selected_distillation_variant": "hard_label_only" if promoted else "hard_label_context_arm",
                "promotion_status": "development_promoted_pending_fresh_oos" if promoted else "not_promoted",
                "j4_best_config": "cfg",
                "j4_best_median_episode_delta_hr30": 0.01,
                "j4_best_seed_pass_rate": 2 / 3 if promoted else 0.0,
                "fresh_oos_status": "pending_later_labelled_interval",
            }
        ]
    ).to_csv(path / "j4_j5_contextual_meta_freeze_decisions.csv", index=False)
    pd.DataFrame(
        [
            {
                "head": head,
                "config_id": "cfg",
                "regime": "moderate",
                "seed_count": 3,
                "seed_pass_rate": 2 / 3 if promoted else 0.0,
                "median_episode_delta_hr30": 0.01,
                "q25_episode_delta_hr30": 0.0,
                "median_delta_hr30": 0.002,
                "median_delta_ndcg": 0.001,
                "median_net_correct": 3,
                "median_delta_hr10": 0.0,
                "median_delta_hr20": 0.0,
                "min_leaf_count_min": 100,
                "context_split_share_mean": 0.01,
                "context_gain_share_mean": 0.001,
                "config_promoted": promoted,
            }
        ]
    ).to_csv(path / "j4_contextual_meta_config_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "head": head,
                "selected_feature_arm": "B_current_plus_model_state",
                "selection_source": "source.csv",
                "selection_objective": "directional",
            }
        ]
    ).to_csv(path / "j4_j5_contextual_meta_feature_arm_freeze.csv", index=False)
    pd.DataFrame(
        {
            "head": [head, head],
            "arm": ["J4_cfg_seed29", "J4_cfg_seed29"],
            "timestamp": ["2026-06-01 00:00:00+00:00", "2026-06-02 00:00:00+00:00"],
        }
    ).to_csv(path / "j4_j5_contextual_meta_directional_timestamp_metrics.csv", index=False)
    pd.DataFrame({"head": [head] * j5_rows, "arm": ["J5_cfg_hard_label_only"] * j5_rows}).to_csv(
        path / "j5_contextual_meta_distillation_summary.csv", index=False
    )
    (path / "j4_j5_contextual_meta_requirement_audit.json").write_text(json.dumps({"status": "passed", "items": []}))
    (path / "run_config.json").write_text(
        json.dumps(
            {
                "baseline_artifact_dir": "data_perp/artifacts/baseline",
                "rank_threshold": 0.70,
                "directional_hr_tolerance": 0.001,
                "min_seed_pass_rate": 2 / 3,
                "j4_seeds": [29, 31, 37],
                "outer_folds": 5,
                "max_j4_configs": 10,
                "max_train_rows": 60000,
            }
        )
    )


def test_manifest_records_explicit_freeze_contract(tmp_path: Path) -> None:
    dirs = []
    for head in mod.EXPECTED_HEADS:
        d = tmp_path / head
        _write_minimal_artifact(d, head)
        dirs.append(d)

    manifest, top_configs, audit = mod.build_manifest(dirs)

    assert audit["status"] == "passed"
    assert set(manifest["head"]) == set(mod.EXPECTED_HEADS)
    assert manifest["baseline_artifact_dir"].eq("data_perp/artifacts/baseline").all()
    assert manifest["model_contract"].str.contains("unchanged_y_bin").all()
    assert manifest["sample_weight_contract"].eq("ordinary_bce_no_top30_reweighting").all()
    assert manifest["selected_capacity_config"].eq("none_retain_context_arm").all()
    assert manifest["fresh_oos_status"].eq("pending_later_labelled_interval").all()
    assert not top_configs.empty


def test_audit_rejects_j5_rows_without_promoted_j4(tmp_path: Path) -> None:
    dirs = []
    for idx, head in enumerate(mod.EXPECTED_HEADS):
        d = tmp_path / head
        _write_minimal_artifact(d, head, promoted=False, j5_rows=1 if idx == 0 else 0)
        dirs.append(d)

    _, _, audit = mod.build_manifest(dirs)

    assert audit["status"] == "failed"
    item = next(x for x in audit["items"] if x["requirement"] == "no_j5_without_promoted_j4")
    assert item["status"] == "failed"


def test_effective_oos_boundary_uses_latest_timestamp() -> None:
    assert (
        mod._max_iso_timestamp("2026-06-10T13:00:00+00:00", "2026-06-15T04:00:00+00:00")
        == "2026-06-15T04:00:00+00:00"
    )
    assert mod._max_iso_timestamp("", None) == ""
