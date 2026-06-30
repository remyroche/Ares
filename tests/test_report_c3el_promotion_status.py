import json

import pandas as pd

from scripts.report_c3el_promotion_status import (
    _active_heads_from_manifest,
    _applied_heads_from_manifest,
    _evidence_reading_lines,
    build_report,
)


def _write_candidate(run_dir, *, active_heads, interventions, candidate_net_pnl, candidate_hr, candidate_full_sl):
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "active_heads": active_heads,
                "interventions": interventions,
                "start": "2026-05-29T00:00:00+00:00",
                "end": "2026-06-26T00:00:00+00:00",
            }
        )
    )
    pd.DataFrame(
        [
            {
                "arm": "C0_baseline",
                "trade_count": 100,
                "net_hit_rate_pct": 48.0,
                "net_pnl": 1000.0,
                "full_sl_rate_pct": 35.0,
                "net_ev_bps_turnover": 100.0,
            },
            {
                "arm": "C3el_head_native",
                "trade_count": 90,
                "net_hit_rate_pct": candidate_hr,
                "net_pnl": candidate_net_pnl,
                "full_sl_rate_pct": candidate_full_sl,
                "net_ev_bps_turnover": 120.0,
            },
        ]
    ).to_csv(run_dir / "overall.csv", index=False)


def _write_candidate_diagnostics(run_dir) -> None:
    pd.DataFrame(
        [
            {
                "arm": "C0_baseline",
                "head": "short_asset",
                "trade_count": 10,
                "net_hit_rate_pct": 40.0,
                "net_pnl": 100.0,
                "full_sl_rate_pct": 50.0,
            },
            {
                "arm": "C3el_head_native",
                "head": "short_asset",
                "trade_count": 9,
                "net_hit_rate_pct": 45.0,
                "net_pnl": 130.0,
                "full_sl_rate_pct": 40.0,
            },
        ]
    ).to_csv(run_dir / "by_head.csv", index=False)
    pd.DataFrame(
        [
            {
                "week_start": "2026-06-01T00:00:00+00:00",
                "head": "short_asset",
                "used_model": True,
                "fallback_used": True,
                "kept_eval_groups": 3,
                "threshold_keep": 2,
                "threshold_value": 15.0,
                "guarded_eval_groups": 1,
                "action_feature_min_guarded_eval_groups": 0,
            },
            {
                "week_start": "2026-06-08T00:00:00+00:00",
                "head": "short_asset",
                "used_model": True,
                "fallback_used": False,
                "kept_eval_groups": 2,
                "threshold_keep": 0,
                "threshold_value": 0.0,
                "guarded_eval_groups": 0,
                "action_feature_min_guarded_eval_groups": 1,
            },
        ]
    ).to_csv(run_dir / "head_native_folds.csv", index=False)
    pd.DataFrame(
        [
            {"timestamp": "2026-06-01T00:00:00+00:00", "strategy_id": "short_asset_a", "multiplier": 0.0},
            {"timestamp": "2026-06-01T01:00:00+00:00", "strategy_id": "short_asset_a", "multiplier": 1.0},
            {"timestamp": "2026-06-01T02:00:00+00:00", "strategy_id": "short_asset_a", "multiplier": 0.5},
        ]
    ).to_csv(run_dir / "head_native_size_schedule.csv", index=False)
    pd.DataFrame(
        [
            {
                "week_start": "2026-06-01T00:00:00+00:00",
                "head": "short_asset",
                "threshold": 0.7,
                "min_pred_delta": 0.0,
                "keep": 2,
                "value": 15.0,
                "eligible": True,
            },
            {
                "week_start": "2026-06-08T00:00:00+00:00",
                "head": "short_asset",
                "threshold": 0.8,
                "min_pred_delta": 0.0,
                "keep": 1,
                "value": -5.0,
                "eligible": True,
            },
        ]
    ).to_csv(run_dir / "head_native_threshold_trials.csv", index=False)


def test_build_report_marks_replay_validated_research_candidate(tmp_path):
    support = tmp_path / "support.csv"
    pd.DataFrame(
        [
            {
                "head": "short_asset",
                "status": "research_candidate",
                "positive_e50_groups": 46,
                "positive_e50_weeks": 9,
            }
        ]
    ).to_csv(support, index=False)
    run_dir = tmp_path / "short_asset"
    _write_candidate(
        run_dir,
        active_heads=["short_asset"],
        interventions=25,
        candidate_net_pnl=1100.0,
        candidate_hr=49.0,
        candidate_full_sl=34.0,
    )

    report = build_report(
        [("short_asset_default", run_dir)],
        support_decision=support,
        min_net_pnl_delta=0.0,
        min_hr_delta_pp=0.0,
        max_full_sl_delta_pp=0.0,
    )

    assert report.loc[0, "disposition"] == "replay_validated_research"
    assert report.loc[0, "delta_net_pnl"] == 100.0
    assert bool(report.loc[0, "allow_monitored_replay"])
    assert not bool(report.loc[0, "allow_production"])
    assert report.loc[0, "gate_reason"] == "replay_passed_but_support_is_research_level"


def test_build_report_rejects_negative_replay_even_with_production_support(tmp_path):
    support = tmp_path / "support.csv"
    pd.DataFrame(
        [
            {
                "head": "short_asset",
                "status": "production_candidate",
                "positive_e50_groups": 80,
                "positive_e50_weeks": 4,
            }
        ]
    ).to_csv(support, index=False)
    run_dir = tmp_path / "bad"
    _write_candidate(
        run_dir,
        active_heads=["short_asset"],
        interventions=10,
        candidate_net_pnl=900.0,
        candidate_hr=47.0,
        candidate_full_sl=36.0,
    )

    report = build_report(
        [("bad_candidate", run_dir)],
        support_decision=support,
        min_net_pnl_delta=0.0,
        min_hr_delta_pp=0.0,
        max_full_sl_delta_pp=0.0,
    )

    assert report.loc[0, "disposition"] == "reject_replay"
    assert not bool(report.loc[0, "allow_monitored_replay"])
    assert not bool(report.loc[0, "allow_production"])


def test_build_report_allows_production_only_when_replay_and_support_pass(tmp_path):
    support = tmp_path / "support.csv"
    pd.DataFrame(
        [
            {
                "head": "short_asset",
                "status": "production_candidate",
                "positive_e50_groups": 80,
                "positive_e50_weeks": 4,
            }
        ]
    ).to_csv(support, index=False)
    run_dir = tmp_path / "good"
    _write_candidate(
        run_dir,
        active_heads=["short_asset"],
        interventions=10,
        candidate_net_pnl=1200.0,
        candidate_hr=50.0,
        candidate_full_sl=33.0,
    )

    report = build_report(
        [("good_candidate", run_dir)],
        support_decision=support,
        min_net_pnl_delta=0.0,
        min_hr_delta_pp=0.0,
        max_full_sl_delta_pp=0.0,
    )

    assert report.loc[0, "disposition"] == "promotion_candidate"
    assert bool(report.loc[0, "allow_monitored_replay"])
    assert bool(report.loc[0, "allow_production"])
    assert report.loc[0, "gate_reason"] == "replay_passed_and_production_support"


def test_build_report_blocks_production_when_readiness_has_no_forward_targets(tmp_path):
    support = tmp_path / "support.csv"
    readiness = tmp_path / "readiness.json"
    pd.DataFrame(
        [
            {
                "head": "short_asset",
                "status": "production_candidate",
                "positive_e50_groups": 80,
                "positive_e50_weeks": 4,
            }
        ]
    ).to_csv(support, index=False)
    readiness.write_text(
        json.dumps(
            {
                "decision": "monitor_only_wait_for_forward_recurrence",
                "unlabeled_target_rows": 0,
                "robust_unlabeled_target_rows": 0,
                "postjun_preferred_firings": 0,
            }
        )
    )
    run_dir = tmp_path / "good"
    _write_candidate(
        run_dir,
        active_heads=["short_asset"],
        interventions=10,
        candidate_net_pnl=1200.0,
        candidate_hr=50.0,
        candidate_full_sl=33.0,
    )

    report = build_report(
        [("good_candidate", run_dir)],
        support_decision=support,
        readiness_manifest=readiness,
        min_net_pnl_delta=0.0,
        min_hr_delta_pp=0.0,
        max_full_sl_delta_pp=0.0,
    )

    row = report.iloc[0]
    assert row["disposition"] == "promotion_candidate"
    assert bool(row["allow_monitored_replay"])
    assert not bool(row["allow_production"])
    assert row["readiness_decision"] == "monitor_only_wait_for_forward_recurrence"
    assert row["readiness_unlabeled_target_rows"] == 0
    assert not bool(row["allow_new_exact_state_replay"])
    assert bool(row["production_blocked_by_readiness"])
    assert row["readiness_blocker"] == "forward_recurrence_missing;no_unique_forward_target_rows"
    assert "forward_recurrence_missing" in row["gate_reason"]


def test_build_report_downgrades_fallback_only_head_native_candidate(tmp_path):
    support = tmp_path / "support.csv"
    pd.DataFrame(
        [
            {
                "head": "short_asset",
                "status": "production_candidate",
                "positive_e50_groups": 80,
                "positive_e50_weeks": 4,
            }
        ]
    ).to_csv(support, index=False)
    run_dir = tmp_path / "fallback_only"
    _write_candidate(
        run_dir,
        active_heads=["short_asset"],
        interventions=10,
        candidate_net_pnl=1200.0,
        candidate_hr=50.0,
        candidate_full_sl=33.0,
    )
    pd.DataFrame(
        [
            {
                "week_start": "2026-06-01T00:00:00+00:00",
                "head": "short_asset",
                "used_model": True,
                "fallback_used": True,
                "kept_eval_groups": 3,
                "threshold_keep": 0,
                "threshold_value": 0.0,
                "guarded_eval_groups": 0,
                "action_feature_min_guarded_eval_groups": 0,
            },
            {
                "week_start": "2026-06-08T00:00:00+00:00",
                "head": "short_asset",
                "used_model": True,
                "fallback_used": True,
                "kept_eval_groups": 2,
                "threshold_keep": 0,
                "threshold_value": 0.0,
                "guarded_eval_groups": 0,
                "action_feature_min_guarded_eval_groups": 0,
            },
        ]
    ).to_csv(run_dir / "head_native_folds.csv", index=False)
    pd.DataFrame(
        [
            {
                "week_start": "2026-06-01T00:00:00+00:00",
                "head": "short_asset",
                "threshold": 0.8,
                "min_pred_delta": 320.0,
                "keep": 2,
                "value": -50.0,
                "eligible": True,
            },
        ]
    ).to_csv(run_dir / "head_native_threshold_trials.csv", index=False)

    report = build_report(
        [("fallback_only", run_dir)],
        support_decision=support,
        min_net_pnl_delta=0.0,
        min_hr_delta_pp=0.0,
        max_full_sl_delta_pp=0.0,
    )

    row = report.iloc[0]
    assert row["selection_evidence_status"] == "fallback_only"
    assert row["selection_evidence_blocker"] == "no_positive_holdout_threshold_trial"
    assert bool(row["threshold_trial_file_present"])
    assert row["threshold_trial_positive_count"] == 0
    assert row["threshold_trial_best_value"] == -50.0
    assert row["disposition"] == "replay_validated_research"
    assert bool(row["allow_monitored_replay"])
    assert not bool(row["allow_production"])


def test_build_report_prefers_simpler_recent_support_default(tmp_path):
    support = tmp_path / "support.csv"
    pd.DataFrame(
        [
            {
                "head": "short_asset",
                "status": "research_candidate",
                "positive_e50_groups": 46,
                "positive_e50_weeks": 9,
                "recent_positive_e50_groups": 6,
                "recent_positive_e50_weeks": 2,
                "support_blocker": "need_14_more_e50_groups",
            },
            {
                "head": "short_boll",
                "status": "research_candidate",
                "positive_e50_groups": 55,
                "positive_e50_weeks": 6,
                "recent_positive_e50_groups": 0,
                "recent_positive_e50_weeks": 0,
                "support_blocker": "need_5_more_e50_groups;no_recent_e50_positive_groups",
            },
        ]
    ).to_csv(support, index=False)
    short_asset = tmp_path / "short_asset"
    combo = tmp_path / "combo"
    _write_candidate(
        short_asset,
        active_heads=["short_asset"],
        interventions=25,
        candidate_net_pnl=1100.0,
        candidate_hr=49.0,
        candidate_full_sl=34.0,
    )
    _write_candidate(
        combo,
        active_heads=["short_asset", "short_boll"],
        interventions=26,
        candidate_net_pnl=1120.0,
        candidate_hr=49.0,
        candidate_full_sl=34.0,
    )

    report = build_report(
        [("short_asset_default", short_asset), ("short_asset_plus_shortboll", combo)],
        support_decision=support,
        min_net_pnl_delta=0.0,
        min_hr_delta_pp=0.0,
        max_full_sl_delta_pp=0.0,
    )

    roles = dict(zip(report["candidate"], report["candidate_role"]))
    assert roles["short_asset_default"] == "monitored_default"
    assert roles["short_asset_plus_shortboll"] == "monitor_research_stale_support"
    assert report["candidate"].iloc[0] == "short_asset_default"


def test_build_report_includes_head_native_diagnostics(tmp_path):
    support = tmp_path / "support.csv"
    pd.DataFrame(
        [
            {
                "head": "short_asset",
                "status": "research_candidate",
                "positive_e50_groups": 46,
                "positive_e50_weeks": 9,
                "recent_positive_e50_groups": 6,
                "recent_positive_e50_weeks": 2,
                "support_blocker": "need_14_more_e50_groups",
            },
        ]
    ).to_csv(support, index=False)
    run_dir = tmp_path / "short_asset"
    _write_candidate(
        run_dir,
        active_heads=["short_asset"],
        interventions=5,
        candidate_net_pnl=1200.0,
        candidate_hr=50.0,
        candidate_full_sl=33.0,
    )
    _write_candidate_diagnostics(run_dir)

    report = build_report(
        [("short_asset_default", run_dir)],
        support_decision=support,
        min_net_pnl_delta=0.0,
        min_hr_delta_pp=0.0,
        max_full_sl_delta_pp=0.0,
    )

    row = report.iloc[0]
    assert row["head_cut_counts"] == '{"short_asset":2}'
    assert row["head_delta_net_pnl"] == '{"short_asset":30.0}'
    assert row["head_delta_hr_pp"] == '{"short_asset":5.0}'
    assert row["head_delta_full_sl_pp"] == '{"short_asset":-10.0}'
    assert row["fallback_used_week_count"] == 1
    assert row["used_model_week_count"] == 2
    assert row["fallback_used_week_rate"] == 0.5
    assert row["kept_eval_groups"] == 5
    assert row["threshold_keep_sum"] == 2
    assert row["threshold_value_sum"] == 15.0
    assert row["positive_threshold_week_count"] == 1
    assert bool(row["threshold_trial_file_present"])
    assert row["threshold_trial_eligible_count"] == 2
    assert row["threshold_trial_positive_count"] == 1
    assert row["threshold_trial_best_value"] == 15.0
    assert row["selection_evidence_status"] == "holdout_positive"
    assert row["selection_evidence_blocker"] == "none"
    assert row["guarded_eval_groups"] == 1
    assert row["action_feature_min_guarded_eval_groups"] == 1


def test_head_native_manifest_without_active_heads_resolves_effective_heads() -> None:
    assert _active_heads_from_manifest(
        {
            "c3el_contract": "head_native",
            "active_heads": [],
            "active_head_configs": {
                "short_asset": {"threshold_grid": [0.8]},
                "short_boll": {"threshold_grid": [0.2]},
            },
        }
    ) == ["short_asset", "short_boll"]


def test_head_native_manifest_without_configs_defaults_to_all_heads() -> None:
    assert _active_heads_from_manifest({"c3el_contract": "head_native", "active_heads": []}) == [
        "long_bars",
        "long_dist",
        "short_asset",
        "short_boll",
    ]


def test_build_report_uses_selected_heads_for_support_gates(tmp_path):
    support = tmp_path / "support.csv"
    pd.DataFrame(
        [
            {
                "head": "short_asset",
                "status": "production_candidate",
                "positive_e50_groups": 80,
                "positive_e50_weeks": 4,
            },
            {
                "head": "short_boll",
                "status": "insufficient_support",
                "positive_e50_groups": 1,
                "positive_e50_weeks": 1,
                "support_blocker": "need_more_e50_groups",
            },
        ]
    ).to_csv(support, index=False)
    run_dir = tmp_path / "scored_two_applied_one"
    _write_candidate(
        run_dir,
        active_heads=["short_asset", "short_boll"],
        interventions=10,
        candidate_net_pnl=1200.0,
        candidate_hr=50.0,
        candidate_full_sl=33.0,
    )
    manifest = json.loads((run_dir / "manifest.json").read_text())
    manifest["c3el_contract"] = "head_native"
    manifest["selected_heads"] = ["short_asset"]
    (run_dir / "manifest.json").write_text(json.dumps(manifest))

    report = build_report(
        [("selected_short_asset", run_dir)],
        support_decision=support,
        min_net_pnl_delta=0.0,
        min_hr_delta_pp=0.0,
        max_full_sl_delta_pp=0.0,
    )

    row = report.iloc[0]
    assert row["scored_heads"] == "short_asset,short_boll"
    assert row["selected_heads"] == "short_asset"
    assert row["active_heads"] == "short_asset"
    assert row["support_status"] == "production_candidate"
    assert row["disposition"] == "promotion_candidate"


def test_applied_heads_from_manifest_falls_back_to_scored_heads_for_legacy_runs() -> None:
    assert _applied_heads_from_manifest({}, ["short_asset", "short_boll"]) == ["short_asset", "short_boll"]
    assert _applied_heads_from_manifest({"selected_heads": []}, ["short_asset", "short_boll"]) == []


def test_evidence_reading_blocks_replay_with_zero_targets_and_fallback_only(tmp_path):
    support = tmp_path / "support.csv"
    readiness = tmp_path / "readiness.json"
    pd.DataFrame(
        [
            {
                "head": "short_asset",
                "status": "research_candidate",
                "positive_e50_groups": 46,
                "positive_e50_weeks": 9,
                "recent_positive_e50_groups": 6,
                "recent_positive_e50_weeks": 2,
                "support_blocker": "need_14_more_e50_groups",
            },
        ]
    ).to_csv(support, index=False)
    readiness.write_text(
        json.dumps(
            {
                "decision": "monitor_only_wait_for_forward_recurrence",
                "unlabeled_target_rows": 0,
                "robust_unlabeled_target_rows": 0,
                "postjun_preferred_firings": 0,
            }
        )
    )
    run_dir = tmp_path / "fallback_only"
    _write_candidate(
        run_dir,
        active_heads=["short_asset"],
        interventions=10,
        candidate_net_pnl=1200.0,
        candidate_hr=50.0,
        candidate_full_sl=33.0,
    )
    pd.DataFrame(
        [
            {
                "week_start": "2026-06-01T00:00:00+00:00",
                "head": "short_asset",
                "used_model": True,
                "fallback_used": True,
                "kept_eval_groups": 3,
                "threshold_keep": 0,
                "threshold_value": 0.0,
            },
        ]
    ).to_csv(run_dir / "head_native_folds.csv", index=False)

    report = build_report(
        [("fallback_only", run_dir)],
        support_decision=support,
        readiness_manifest=readiness,
        min_net_pnl_delta=0.0,
        min_hr_delta_pp=0.0,
        max_full_sl_delta_pp=0.0,
    )
    text = "\n".join(_evidence_reading_lines(report))

    assert "Do not run another exact-state replay yet" in text
    assert "fallback-only replay gains stay research-level" in text
    assert "forward_recurrence_missing;no_unique_forward_target_rows" in text
