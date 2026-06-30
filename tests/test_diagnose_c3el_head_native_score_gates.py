import pandas as pd

from scripts.diagnose_c3el_head_native_score_gates import build_report


def _write_run(tmp_path, scores: list[dict], folds: list[dict]):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(scores).to_csv(run_dir / "head_native_group_scores.csv", index=False)
    pd.DataFrame(folds).to_csv(run_dir / "head_native_folds.csv", index=False)
    return run_dir


def _base_fold(**overrides):
    row = {
        "head": "short_boll",
        "week_start": "2026-06-22T00:00:00+00:00",
        "threshold": 0.5,
        "effective_min_pred_delta": 10.0,
        "max_eval_keep": 2,
        "fallback_used": False,
        "eval_groups": 5,
        "kept_eval_groups": 0,
    }
    row.update(overrides)
    return row


def _score(idx: int, *, p: float, delta: float, keep: bool, guarded: bool = False, head: str = "short_boll"):
    return {
        "head": head,
        "week_start": "2026-06-22T00:00:00+00:00",
        "timestamp": f"2026-06-22T0{idx}:00:00+00:00",
        "strategy_id": f"strategy_{idx}",
        "p_intervene": p,
        "pred_action_delta_J": delta,
        "gate_keep": keep,
        "guard_action_feature_min": guarded,
    }


def test_build_report_detects_feature_guard_blocking_score_candidates(tmp_path):
    run_dir = _write_run(
        tmp_path,
        [
            _score(0, p=0.8, delta=30.0, keep=False, guarded=True),
            _score(1, p=0.7, delta=20.0, keep=False, guarded=True),
            _score(2, p=0.1, delta=5.0, keep=False, guarded=False),
        ],
        [_base_fold(max_eval_keep=5)],
    )

    report = build_report(run_dir)
    week = report.loc[report["week_start"].ne("ALL")].iloc[0]
    head = report.loc[report["week_start"].eq("ALL")].iloc[0]

    assert week["diagnosis"] == "feature_guard_blocks_score_candidates"
    assert head["diagnosis"] == "feature_guard_blocks_score_candidates"
    assert week["score_eligible_groups"] == 2
    assert week["guard_action_feature_min_groups"] == 2
    assert week["gate_kept_groups"] == 0


def test_build_report_detects_cap_limited_selection(tmp_path):
    run_dir = _write_run(
        tmp_path,
        [
            _score(0, p=0.9, delta=50.0, keep=True),
            _score(1, p=0.8, delta=40.0, keep=True),
            _score(2, p=0.7, delta=30.0, keep=False),
            _score(3, p=0.6, delta=20.0, keep=False),
        ],
        [_base_fold(max_eval_keep=2)],
    )

    report = build_report(run_dir)
    week = report.loc[report["week_start"].ne("ALL")].iloc[0]

    assert week["diagnosis"] == "cap_limited"
    assert week["score_eligible_groups"] == 4
    assert week["gate_kept_groups"] == 2
    assert week["max_eval_keep"] == 2


def test_build_report_detects_empty_score_gate(tmp_path):
    run_dir = _write_run(
        tmp_path,
        [
            _score(0, p=0.3, delta=50.0, keep=False),
            _score(1, p=0.8, delta=5.0, keep=False),
        ],
        [_base_fold()],
    )

    report = build_report(run_dir)
    week = report.loc[report["week_start"].ne("ALL")].iloc[0]

    assert week["diagnosis"] == "score_gate_empty"
    assert week["score_eligible_groups"] == 0
    assert week["gate_kept_groups"] == 0
