import pandas as pd

from scripts.report_c3el_current_readiness import build_readiness


def test_build_readiness_marks_monitor_only_until_forward_recurrence() -> None:
    bucket_summary = pd.DataFrame(
        {
            "bucket": ["p80_d320", "p80_d250_320"],
            "rows": [28, 37],
            "pos_share": [0.64, 0.45],
            "sum_delta_full_J": [4603.0, -7582.0],
        }
    )
    rule_summary = pd.DataFrame(
        {
            "rule": [
                "strict_p80_d320",
                "strict__cooldown_count_lte_38_5",
                "strict__cooldown_count_lte_38_5__open_or_cooldown_share_lte_0_3949",
                "strict__at_least_4_conditions",
            ],
            "rows": [28, 21, 12, 9],
            "day_count": [9, 9, 6, 6],
            "positive_share": [0.64, 0.81, 1.0, 1.0],
            "positive_day_share": [0.56, 0.78, 1.0, 1.0],
            "sum_delta_full_J": [4603.0, 9335.0, 8195.0, 6551.0],
            "worst_delta_full_J": [-1421.0, -687.0, 130.0, 130.0],
            "passes_min_rows": [True, True, True, "False"],
        }
    )
    monitor_counts = pd.DataFrame(
        {
            "panel": ["last4w", "last4w", "postjun26"],
            "rule": [
                "rule_p80_d320_cooldown_lte_38_5",
                "rule_p80_d320_cooldown_lte_38_5_open_or_cooldown_share_lte_0_3949",
                "rule_p80_d320_cooldown_lte_38_5",
            ],
            "rows": [21, 12, 0],
        }
    )
    target_backlog = pd.DataFrame(
        {
            "queue": ["last4w", "last4w_robust"],
            "rule": [
                "rule_p80_d320_cooldown_lte_38_5",
                "rule_p80_d320_cooldown_lte_38_5_open_or_cooldown_share_lte_0_3949",
            ],
            "candidate_pool_rows": [0, 0],
            "target_rows": [0, 0],
        }
    )

    gates, payload = build_readiness(
        bucket_summary=bucket_summary,
        rule_summary=rule_summary,
        monitor_counts=monitor_counts,
        target_backlog=target_backlog,
    )

    statuses = dict(zip(gates["gate"], gates["status"], strict=True))
    assert statuses["strict_exact_state_evidence"] == "pass"
    assert statuses["broadening_rejected"] == "pass"
    assert statuses["preferred_rule_lift"] == "pass"
    assert statuses["robust_rule_precision"] == "pass"
    assert statuses["ultra_conservative_subset"] == "watch"
    assert statuses["forward_recurrence"] == "waiting"
    assert statuses["unlabeled_target_backlog"] == "waiting"
    assert statuses["production_promotion"] == "fail"
    assert payload["decision"] == "monitor_only_wait_for_forward_recurrence"
    assert payload["last4w_robust_firings"] == 12
    assert payload["robust_unlabeled_target_rows"] == 0


def test_build_readiness_uses_unique_target_backlog_actions(tmp_path) -> None:
    ts = pd.Timestamp("2026-06-15 00:00:00", tz="UTC")
    target_a = tmp_path / "a_targets.csv"
    target_b = tmp_path / "b_targets.csv"
    pool_a = tmp_path / "a_pool.csv"
    pool_b = tmp_path / "b_pool.csv"
    duplicate = {
        "timestamp": ts,
        "strategy_id": "short_asset_alpha",
        "action_family": "size",
        "action_value": 0.0,
    }
    extra = {
        "timestamp": ts + pd.Timedelta(hours=1),
        "strategy_id": "short_asset_alpha",
        "action_family": "size",
        "action_value": 0.0,
    }
    pd.DataFrame([duplicate]).to_csv(target_a, index=False)
    pd.DataFrame([duplicate, extra]).to_csv(target_b, index=False)
    pd.DataFrame([duplicate]).to_csv(pool_a, index=False)
    pd.DataFrame([duplicate, extra]).to_csv(pool_b, index=False)

    bucket_summary = pd.DataFrame(
        {
            "bucket": ["p80_d320", "p80_d250_320"],
            "rows": [28, 37],
            "pos_share": [0.64, 0.45],
            "sum_delta_full_J": [4603.0, -7582.0],
        }
    )
    rule_summary = pd.DataFrame(
        {
            "rule": [
                "strict_p80_d320",
                "strict__cooldown_count_lte_38_5",
                "strict__cooldown_count_lte_38_5__open_or_cooldown_share_lte_0_3949",
                "strict__at_least_4_conditions",
            ],
            "rows": [28, 21, 12, 9],
            "day_count": [9, 9, 6, 6],
            "positive_share": [0.64, 0.81, 1.0, 1.0],
            "positive_day_share": [0.56, 0.78, 1.0, 1.0],
            "sum_delta_full_J": [4603.0, 9335.0, 8195.0, 6551.0],
            "worst_delta_full_J": [-1421.0, -687.0, 130.0, 130.0],
            "passes_min_rows": [True, True, True, False],
        }
    )
    monitor_counts = pd.DataFrame(
        {
            "panel": ["last4w", "last4w", "postjun26"],
            "rule": [
                "rule_p80_d320_cooldown_lte_38_5",
                "rule_p80_d320_cooldown_lte_38_5_open_or_cooldown_share_lte_0_3949",
                "rule_p80_d320_cooldown_lte_38_5",
            ],
            "rows": [21, 12, 1],
        }
    )
    target_backlog = pd.DataFrame(
        {
            "queue": ["cooldown", "robust"],
            "rule": [
                "rule_p80_d320_cooldown_lte_38_5",
                "rule_p80_d320_cooldown_lte_38_5_open_or_cooldown_share_lte_0_3949",
            ],
            "candidate_pool_rows": [1, 2],
            "target_rows": [1, 2],
            "candidate_pool_path": [str(pool_a), str(pool_b)],
            "target_actions_path": [str(target_a), str(target_b)],
        }
    )

    gates, payload = build_readiness(
        bucket_summary=bucket_summary,
        rule_summary=rule_summary,
        monitor_counts=monitor_counts,
        target_backlog=target_backlog,
    )

    statuses = dict(zip(gates["gate"], gates["status"], strict=True))
    backlog_gate = gates.loc[gates["gate"].eq("unlabeled_target_backlog")].iloc[0]
    assert statuses["unlabeled_target_backlog"] == "ready"
    assert "candidate_pool_rows=2" in backlog_gate["evidence"]
    assert "selected target_rows=2" in backlog_gate["evidence"]
    assert payload["unlabeled_target_rows"] == 2
    assert payload["robust_unlabeled_target_rows"] == 2
