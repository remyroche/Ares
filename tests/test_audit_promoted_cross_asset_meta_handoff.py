from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_promoted_cross_asset_meta_handoff import run_audit  # noqa: E402


def _selector(
    *,
    exec10: float,
    clean10: float,
    bad10: float,
    recall10: float,
    exec30: float,
    clean30: float,
    bad30: float,
    recall30: float,
    auc: float,
    ap: float,
) -> dict:
    return {
        "selector": "meta_clean_exec",
        "mean_keep010_exec_margin": exec10,
        "mean_keep010_clean_exec_precision": clean10,
        "mean_keep010_full_path_bad_mae": bad10,
        "mean_keep010_timeout": 0.01,
        "mean_keep010_oracle_recall": recall10,
        "mean_keep030_exec_margin": exec30,
        "mean_keep030_clean_exec_precision": clean30,
        "mean_keep030_full_path_bad_mae": bad30,
        "mean_keep030_timeout": 0.01,
        "mean_keep030_oracle_recall": recall30,
        "mean_auc_clean_exec": auc,
        "mean_ap_clean_exec": ap,
    }


def _write_smoke(root: Path, selector: dict, *, forbidden_feature: bool = False) -> None:
    root.mkdir(parents=True)
    payload = {
        "best_selector": selector,
        "best_threshold_policy": {
            "selector": "meta_clean_exec",
            "policy_id": "clean_ge_0.55",
            "threshold_policy_status": "diagnostic_or_fail",
        },
    }
    (root / "manifest.json").write_text(json.dumps(payload))
    features = [
        {"test_month": "2026-06", "model": "clean_exec", "feature": "cross_lgbm_bad_mae_score", "importance": 12.0},
        {"test_month": "2026-06", "model": "bad_path", "feature": "cross_lgbm_timeout_score", "importance": 8.0},
        {
            "test_month": "2026-06",
            "model": "clean_exec",
            "feature": "long_bad_path_label" if forbidden_feature else "regime_clean_exec_score",
            "importance": 5.0,
        },
    ]
    pd.DataFrame(features).to_csv(root / "s52_train_meta_regime_handoff_smoke_feature_importance.csv", index=False)


def _write_contract(root: Path, *, no_backfill: bool = True) -> None:
    root.mkdir(parents=True)
    payload = {
        "promoted_cross_asset_representation": {
            "preferred_variant": "m1b_cross_lgbm_risk_only_meta",
            "promoted_columns": [
                "cross_lgbm_bad_mae_score",
                "cross_lgbm_timeout_score",
            ],
            "rows_with_all_promoted_columns": 100,
            "coverage_all_promoted_columns": 0.25,
            "no_in_sample_backfill": no_backfill,
        }
    }
    (root / "train_meta_regime_handoff_contract.json").write_text(json.dumps(payload))


def _write_promotion(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "candidate_features_available",
                "promote_to_deeper_meta_eval": [{"variant": "m1b_cross_lgbm_risk_only_meta"}],
            }
        )
    )


def test_promoted_cross_asset_meta_handoff_audit_conditional_pass(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    promoted = tmp_path / "promoted_smoke"
    handoff = tmp_path / "handoff"
    promotion = tmp_path / "promotion.json"
    _write_smoke(
        baseline,
        _selector(
            exec10=0.010,
            clean10=0.63,
            bad10=0.56,
            recall10=0.19,
            exec30=0.0070,
            clean30=0.60,
            bad30=0.55,
            recall30=0.42,
            auc=0.60,
            ap=0.56,
        ),
    )
    _write_smoke(
        promoted,
        _selector(
            exec10=0.011,
            clean10=0.65,
            bad10=0.52,
            recall10=0.185,
            exec30=0.0074,
            clean30=0.61,
            bad30=0.57,
            recall30=0.43,
            auc=0.62,
            ap=0.58,
        ),
    )
    _write_contract(handoff)
    _write_promotion(promotion)

    payload = run_audit(
        baseline_smoke_dir=baseline,
        promoted_smoke_dir=promoted,
        promoted_handoff_dir=handoff,
        promotion_json=promotion,
        out_dir=tmp_path / "out",
    )

    assert payload["gate_decision"]["meta_feature_status"] == "conditional_pass_for_deeper_meta_eval"
    assert "top30_bad_mae_nonworse" in payload["gate_decision"]["warning_flags"]
    assert payload["feature_checks"]["promoted_column_use_count"] == 2
    assert Path(payload["outputs"]["markdown"]).exists()
    deltas = pd.read_csv(payload["outputs"]["metric_deltas"])
    top10_bad = deltas.set_index("metric").loc["mean_keep010_full_path_bad_mae", "delta"]
    assert top10_bad < 0.0


def test_promoted_cross_asset_meta_handoff_audit_blocks_forbidden_feature(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    promoted = tmp_path / "promoted_smoke"
    handoff = tmp_path / "handoff"
    promotion = tmp_path / "promotion.json"
    selector = _selector(
        exec10=0.010,
        clean10=0.63,
        bad10=0.56,
        recall10=0.19,
        exec30=0.0070,
        clean30=0.60,
        bad30=0.55,
        recall30=0.42,
        auc=0.60,
        ap=0.56,
    )
    _write_smoke(baseline, selector)
    _write_smoke(promoted, selector | {"mean_keep010_exec_margin": 0.011}, forbidden_feature=True)
    _write_contract(handoff, no_backfill=False)
    _write_promotion(promotion)

    payload = run_audit(
        baseline_smoke_dir=baseline,
        promoted_smoke_dir=promoted,
        promoted_handoff_dir=handoff,
        promotion_json=promotion,
        out_dir=tmp_path / "out",
    )

    assert payload["gate_decision"]["meta_feature_status"] == "blocked"
    assert "feature_no_forbidden_outcomes" in payload["gate_decision"]["failed_checks"]
    assert "handoff_no_in_sample_backfill" in payload["gate_decision"]["failed_checks"]
