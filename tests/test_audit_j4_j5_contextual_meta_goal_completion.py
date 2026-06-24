from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts import audit_j4_j5_contextual_meta_goal_completion as mod


def _write_common(tmp_path: Path, *, ready: bool) -> dict[str, Path]:
    top30 = tmp_path / "top30.json"
    top30.write_text(json.dumps({"status": "passed"}))
    freeze_audit = tmp_path / "freeze_audit.json"
    freeze_audit.write_text(json.dumps({"status": "passed"}))
    readiness_audit = tmp_path / "readiness_audit.json"
    readiness_audit.write_text(json.dumps({"status": "ready" if ready else "not_ready"}))
    eval_audit = tmp_path / "eval_audit.json"
    eval_audit.write_text(json.dumps({"status": "passed" if ready else "not_ready"}))

    freeze = tmp_path / "freeze.csv"
    pd.DataFrame(
        [
            {
                "head": head,
                "baseline_artifact_dir": "data_perp/artifacts/baseline",
                "sample_weight_contract": "ordinary_bce_no_top30_reweighting",
                "j4_seeds": "29,31,37",
                "max_j4_configs": 10,
                "selected_contextual_feature_arm": "B_current_plus_model_state",
                "selected_capacity_config": "none_retain_context_arm",
                "selected_distillation_variant": "hard_label_context_arm",
                "rank_threshold": 0.70,
                "hr10_min_delta": -0.001,
                "hr20_min_delta": -0.001,
                "normal_period_hr30_min_delta": -0.001,
                "ndcg30_min_delta": 0.0,
                "decision": "retain_selected_contextual_feature_arm_j4_not_promoted",
                "j5_rows": 0,
            }
            for head in mod.EXPECTED_HEADS
        ]
    ).to_csv(freeze, index=False)
    readiness = tmp_path / "readiness.csv"
    pd.DataFrame(
        [
            {
                "head": head,
                "ready_for_fresh_oos_confirmation": ready,
                "guarded_fresh_oos_start": "2026-06-16T04:00:00+00:00",
                "label_rows_after_guard": 1000 if ready else 0,
                "candidate_score_rows_after_guard": 1000 if ready else 0,
            }
            for head in mod.EXPECTED_HEADS
        ]
    ).to_csv(readiness, index=False)
    return {
        "top30": top30,
        "freeze": freeze,
        "freeze_audit": freeze_audit,
        "readiness": readiness,
        "readiness_audit": readiness_audit,
        "eval_audit": eval_audit,
    }


def test_goal_audit_blocks_when_fresh_oos_not_ready(tmp_path: Path) -> None:
    paths = _write_common(tmp_path, ready=False)

    table, audit = mod.build_audit(
        top30_audit_path=paths["top30"],
        freeze_manifest_path=paths["freeze"],
        freeze_audit_path=paths["freeze_audit"],
        readiness_path=paths["readiness"],
        readiness_audit_path=paths["readiness_audit"],
        eval_audit_path=paths["eval_audit"],
    )

    assert audit["status"] == "blocked_pending_fresh_oos"
    assert table.loc[table["requirement"].str.contains("Fresh chronological OOS"), "status"].iloc[0] == "blocked_pending_fresh_oos"


def test_goal_audit_completes_when_all_requirements_and_oos_pass(tmp_path: Path) -> None:
    paths = _write_common(tmp_path, ready=True)

    _, audit = mod.build_audit(
        top30_audit_path=paths["top30"],
        freeze_manifest_path=paths["freeze"],
        freeze_audit_path=paths["freeze_audit"],
        readiness_path=paths["readiness"],
        readiness_audit_path=paths["readiness_audit"],
        eval_audit_path=paths["eval_audit"],
    )

    assert audit["status"] == "complete"
