from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.report_s52_meta_handoff_gate3_readiness import run_audit


def _write_variant(root: Path, name: str, offline: pd.DataFrame, clean_extra: dict | None = None) -> None:
    d = root / name
    d.mkdir(parents=True)
    clean_cols = [
        "timestamp",
        "symbol",
        "side_name",
        "month",
        "source_semantic_family",
        "aegmm_cluster",
        "regime_clean_exec_score_bin",
    ]
    clean = offline.rename(columns={"__ts__": "timestamp", "__symbol__": "symbol"})[
        [col for col in clean_cols if col in offline.rename(columns={"__ts__": "timestamp", "__symbol__": "symbol"}).columns]
    ].copy()
    if clean_extra:
        for key, value in clean_extra.items():
            clean[key] = value
    clean.to_parquet(d / "s52_meta_threshold_guarded_candidates.parquet", index=False)
    offline.to_parquet(d / "s52_meta_threshold_guarded_offline_eval_candidates.parquet", index=False)
    (d / "s52_meta_threshold_guarded_leakage_audit.json").write_text(
        json.dumps(
            {
                "clean_handoff_forbidden_columns": [],
                "duplicate_decision_key_rows": 0,
            }
        )
    )


def test_gate3_readiness_blocks_month_bad_mae_and_reports_breakdowns(tmp_path: Path) -> None:
    root = tmp_path / "handoff"
    offline = pd.DataFrame(
        {
            "__ts__": pd.date_range("2026-05-01", periods=6, freq="h").astype(str),
            "__symbol__": ["BTC", "ETH", "SOL", "BTC", "ETH", "SOL"],
            "side_name": ["short", "short", "long", "short", "long", "long"],
            "month": ["2026-05", "2026-05", "2026-05", "2026-06", "2026-06", "2026-06"],
            "source_semantic_family": ["mean_reversion", "mean_reversion", "mixed", "mixed", "mixed", "trend"],
            "aegmm_cluster": ["c0", "c0", "c1", "c1", "c1", "c2"],
            "regime_clean_exec_score_bin": ["q3", "q3", "q2", "q1", "q1", "q2"],
            "regime_dirty_positive_score_bin": ["q1", "q1", "q2", "q3", "q3", "q2"],
            "regime_first_touch_bad_mae_score_bin": ["q1", "q1", "q2", "q3", "q3", "q2"],
            "regime_lgbm_leaf_exec_margin_k4": ["l0", "l0", "l1", "l1", "l1", "l2"],
            "exec_margin": [0.02, 0.01, 0.01, 0.02, 0.01, 0.01],
            "ret_net": [0.024, 0.014, 0.014, 0.024, 0.014, 0.014],
            "u_policy_net": [0.027, 0.017, 0.017, 0.027, 0.017, 0.017],
            "full_path_bad_mae_1r": [0, 0, 0, 1, 1, 0],
            "timeout": [0, 0, 0, 0, 0, 0],
            "clean_exec": [1, 1, 1, 0, 0, 1],
            "dirty_positive": [0, 0, 0, 1, 1, 0],
            "mfe_before_mae_1r": [1, 1, 1, 0, 0, 1],
            "mae_before_mfe_1r": [0, 0, 0, 1, 1, 0],
        }
    )
    _write_variant(root, "candidate", offline)

    out = tmp_path / "out"
    manifest = run_audit(
        handoff_root=root,
        variants=("candidate",),
        out_dir=out,
        max_side_share=0.80,
        min_rows=1,
        min_symbols=1,
    )

    summary = pd.read_csv(manifest["outputs"]["summary"])
    breakdown = pd.read_csv(manifest["outputs"]["breakdown"])
    failures = pd.read_csv(manifest["outputs"]["failures"])

    assert summary.loc[0, "gate3_status"] == "blocked"
    assert "month_bad_mae_bar" in summary.loc[0, "failed_checks"]
    assert "side_source" in set(breakdown["grouping"])
    assert "month_bad_mae_bar" in set(failures["failed_check"])

    override = tmp_path / "override"
    manifest_override = run_audit(
        handoff_root=root,
        variants=("candidate",),
        out_dir=override,
        max_side_share=1.0,
        min_rows=1,
        min_symbols=1,
        allow_bad_mae_pnl_override=True,
        min_override_mean_ret_net=0.005,
        min_override_worst_month_ret_net=0.005,
    )
    override_summary = pd.read_csv(manifest_override["outputs"]["summary"])
    assert override_summary.loc[0, "gate3_status"] == "pnl_override_candidate"
    assert override_summary.loc[0, "path_risk_status"] == "bad_mae_accepted_by_pnl_override"
    assert bool(override_summary.loc[0, "bad_mae_accepted_by_pnl_override"])
    assert "month_bad_mae_bar" in override_summary.loc[0, "bad_mae_only_failures"]


def test_gate3_readiness_detects_clean_handoff_outcome_leakage(tmp_path: Path) -> None:
    root = tmp_path / "handoff"
    offline = pd.DataFrame(
        {
            "__ts__": pd.date_range("2026-05-01", periods=2, freq="h").astype(str),
            "__symbol__": ["BTC", "ETH"],
            "side_name": ["long", "short"],
            "month": ["2026-05", "2026-05"],
            "source_semantic_family": ["trend", "trend"],
            "exec_margin": [0.02, 0.03],
            "ret_net": [0.024, 0.034],
            "u_policy_net": [0.027, 0.037],
            "full_path_bad_mae_1r": [0, 0],
            "timeout": [0, 0],
            "clean_exec": [1, 1],
            "dirty_positive": [0, 0],
        }
    )
    _write_variant(root, "leaky", offline, clean_extra={"exec_margin": 0.01})

    manifest = run_audit(
        handoff_root=root,
        variants=("leaky",),
        out_dir=tmp_path / "out",
        max_side_share=1.0,
        min_rows=1,
        min_symbols=1,
    )
    summary = pd.read_csv(manifest["outputs"]["summary"])

    assert summary.loc[0, "pass_clean_handoff_no_outcomes"] is False or not bool(
        summary.loc[0, "pass_clean_handoff_no_outcomes"]
    )
    assert "clean_handoff_no_outcomes" in summary.loc[0, "failed_checks"]
