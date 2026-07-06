import json
from pathlib import Path

import pandas as pd

from scripts.check_gmm_train_base_meta_readiness import build_readiness_check


def _write_report(root: Path, *, readiness_rows: int = 1, learnability_status: str | None = None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "label_viability_matrix": {"active_rows": 1},
                "train_base_meta_readiness": {
                    "required_next_checks": [
                        "train_base_oof_learnability",
                        "train_meta_oos_profitability",
                        "simple_policy_optimiser_exit_policy",
                        "frozen_threshold_replay",
                        "leakage_and_feature_parity_audit",
                    ],
                },
            }
        )
    )
    pd.DataFrame(
        [
            {
                "cluster_policy": "s14_path_quality_risk_trim_score",
                "top_frac": 0.03,
                "active_label": True,
                "first_failed_gate": "pass",
                "label_viability_score": 100.0,
            },
            {
                "cluster_policy": "s15_side_path_quality_risk_trim_score",
                "top_frac": 0.03,
                "active_label": False,
                "first_failed_gate": "tail_risk",
                "label_viability_score": 83.3,
                "mean_u": 0.003,
                "bad_mae_1r_rate": 0.56,
            },
        ]
    ).to_csv(root / "gmm_label_viability_matrix.csv", index=False)
    readiness = pd.DataFrame(
        [
            {
                "readiness_status": "candidate_for_train_base_meta_smoke",
                "is_final_promotion_ready": False,
                "cluster_policy": "s14_path_quality_risk_trim_score",
                "top_frac": 0.03,
                "labels_path": "labels",
                "feature_dir": "features",
                "feature_list_csv": "features.csv",
                "required_next_checks": (
                    "train_base_oof_learnability;train_meta_oos_profitability;"
                    "simple_policy_optimiser_exit_policy;frozen_threshold_replay;"
                    "leakage_and_feature_parity_audit"
                ),
            }
        ][:readiness_rows]
    )
    readiness.to_csv(root / "gmm_train_meta_readiness.csv", index=False)
    if learnability_status is not None:
        (root / "gmm_train_base_learnability_check.json").write_text(
            json.dumps(
                {
                    "status": learnability_status,
                    "passed_next_check": (
                        "train_base_oof_learnability" if learnability_status == "pass" else None
                    ),
                    "failed_next_check": (
                        None if learnability_status == "pass" else "train_base_oof_learnability"
                    ),
                }
            )
        )


def test_build_readiness_check_accepts_active_s14_candidate(tmp_path: Path) -> None:
    _write_report(tmp_path)

    report = build_readiness_check(tmp_path)

    assert report["status"] == "candidate_for_train_base_meta_smoke"
    assert report["candidate_selectors"] == ["s14_path_quality_risk_trim_score"]
    assert report["final_promotion_ready"] is False
    assert report["pending_next_checks"][0] == "train_base_oof_learnability"
    assert report["errors"] == []
    assert report["comparator_failures"][0]["cluster_policy"].startswith("s15_")


def test_build_readiness_check_advances_after_learnability_pass(tmp_path: Path) -> None:
    _write_report(tmp_path, learnability_status="pass")

    report = build_readiness_check(tmp_path)

    assert report["status"] == "candidate_for_train_meta_profitability_smoke"
    assert report["passed_next_checks"] == ["train_base_oof_learnability"]
    assert "train_base_oof_learnability" not in report["pending_next_checks"]
    assert report["errors"] == []


def test_build_readiness_check_advances_to_meta_path_filter_after_candidate_ready(
    tmp_path: Path,
) -> None:
    _write_report(tmp_path, learnability_status="candidate_for_train_meta_path_filter_smoke")

    report = build_readiness_check(tmp_path)

    assert report["status"] == "candidate_for_train_meta_path_filter_smoke"
    assert report["passed_next_checks"] == ["train_base_oof_learnability"]
    assert "train_base_final_policy_readiness_failed_meta_filter_required" in report["warnings"]
    assert report["errors"] == []


def test_build_readiness_check_blocks_after_learnability_failure(tmp_path: Path) -> None:
    _write_report(tmp_path, learnability_status="fail")

    report = build_readiness_check(tmp_path)

    assert report["status"] == "not_ready_for_train_meta_profitability"
    assert report["failed_next_checks"] == ["train_base_oof_learnability"]
    assert "learnability_check_failed:fail" in report["errors"]


def test_build_readiness_check_rejects_readiness_viability_mismatch(tmp_path: Path) -> None:
    _write_report(tmp_path, readiness_rows=0)

    report = build_readiness_check(tmp_path)

    assert report["status"] == "not_ready"
    assert "readiness_rows_mismatch:0!=1" in report["errors"]
