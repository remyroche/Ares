import pandas as pd

from scripts.diagnose_c3el_threshold_selection import summarise_candidate


def _write_folds(run_dir, *, head="short_asset") -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "week_start": "2026-06-01T00:00:00+00:00",
                "head": head,
                "used_model": True,
                "fallback_used": True,
                "eval_groups": 10,
                "kept_eval_groups": 2,
                "threshold_keep": 0,
                "threshold_value": 0.0,
                "train_groups": 100,
                "train_positive_groups": 4,
                "train_positive_group_rate": 0.04,
            },
            {
                "week_start": "2026-06-08T00:00:00+00:00",
                "head": head,
                "used_model": True,
                "fallback_used": True,
                "eval_groups": 30,
                "kept_eval_groups": 3,
                "threshold_keep": 0,
                "threshold_value": 0.0,
                "train_groups": 110,
                "train_positive_groups": 5,
                "train_positive_group_rate": 0.045,
            },
        ]
    ).to_csv(run_dir / "head_native_folds.csv", index=False)


def test_summarise_candidate_detects_negative_holdout_threshold_trials(tmp_path):
    run_dir = tmp_path / "run"
    _write_folds(run_dir)
    pd.DataFrame(
        [
            {
                "week_start": "2026-06-01T00:00:00+00:00",
                "head": "short_asset",
                "threshold": 0.35,
                "min_pred_delta": 320.0,
                "keep": 2,
                "value": -10.0,
                "eligible": True,
            },
            {
                "week_start": "2026-06-01T00:00:00+00:00",
                "head": "short_asset",
                "threshold": 0.85,
                "min_pred_delta": 0.0,
                "keep": 9,
                "value": -25.0,
                "eligible": True,
            },
        ]
    ).to_csv(run_dir / "head_native_threshold_trials.csv", index=False)

    report = summarise_candidate("demo", run_dir)
    row = report.iloc[0]

    assert row["diagnosis"] == "holdout_selection_negative"
    assert bool(row["threshold_trial_file_present"])
    assert row["threshold_trial_eligible_count"] == 2
    assert row["threshold_trial_positive_count"] == 0
    assert row["threshold_trial_best_value"] == -10.0
    assert row["fallback_used_week_rate"] == 1.0
    assert row["kept_eval_share"] == 0.125


def test_summarise_candidate_detects_missing_threshold_trial_artifact(tmp_path):
    run_dir = tmp_path / "run"
    _write_folds(run_dir, head="short_boll")

    report = summarise_candidate("demo", run_dir)
    row = report.iloc[0]

    assert row["head"] == "short_boll"
    assert row["diagnosis"] == "missing_threshold_trial_artifact"
    assert not bool(row["threshold_trial_file_present"])
    assert row["threshold_trial_eligible_count"] == 0
    assert row["recommendation"] == "rerun_or_materialize_threshold_trials_before_comparing_candidate"
