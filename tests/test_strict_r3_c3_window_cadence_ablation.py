from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts" / "run_strict_r3_c3_window_cadence_ablation.py"
SPEC = importlib.util.spec_from_file_location("strict_r3_c3_window_runner", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_window_geometry_precedes_one_consistent_meta_contract() -> None:
    cutoff = pd.Timestamp("2025-07-01", tz="UTC")
    spec = MODULE.ArmSpec("window_4m", training_months=4, burnin_months=3)
    geometry_start, geometry_end, train_start = MODULE._model_windows(spec, cutoff)
    assert geometry_start == pd.Timestamp("2024-12-01", tz="UTC")
    assert geometry_end == pd.Timestamp("2025-03-01", tz="UTC")
    assert train_start == geometry_end
    assert train_start == pd.Timestamp("2025-03-01", tz="UTC")


def test_weekly_cadence_is_bounded_and_covers_declared_interval() -> None:
    start = pd.Timestamp("2025-01-01", tz="UTC")
    end = pd.Timestamp("2025-03-01", tz="UTC")
    spec = MODULE.ArmSpec(
        "cadence4w", training_months=3, burnin_months=3,
        cadence_weeks=4, schedule="weeks",
    )
    blocks = MODULE._blocks(spec, start, end)
    assert blocks[0][0] == start
    assert blocks[-1][1] == end
    assert all(left < right for left, right in blocks)
    assert all((right - left) <= pd.Timedelta(weeks=4) for left, right in blocks)


def test_admission_warmup_preserves_evaluation_refit_boundary() -> None:
    start = pd.Timestamp("2026-01-01", tz="UTC")
    four_week = MODULE.ArmSpec(
        "cadence4w", training_months=6, burnin_months=3,
        cadence_weeks=4, schedule="weeks",
    )
    eight_week = MODULE.ArmSpec(
        "cadence8w", training_months=6, burnin_months=3,
        cadence_weeks=8, schedule="weeks",
    )
    four_start = MODULE._admission_scoring_start(four_week, start, 84)
    eight_start = MODULE._admission_scoring_start(eight_week, start, 84)
    assert four_start == start - pd.Timedelta(weeks=12)
    assert eight_start == start - pd.Timedelta(weeks=16)
    assert start in {left for left, _ in MODULE._blocks(four_week, four_start, start + pd.Timedelta(days=1))}
    assert start in {left for left, _ in MODULE._blocks(eight_week, eight_start, start + pd.Timedelta(days=1))}


def test_portability_penalises_a_negative_worst_month() -> None:
    stable = MODULE._portability([20.0, 25.0, 30.0])
    fragile = MODULE._portability([-20.0, 25.0, 30.0])
    assert stable[0] > fragile[0]
    assert fragile[3] == -20.0


def test_severe_target_is_frozen_to_h12_tp6_sl4() -> None:
    assert not any(name.startswith("policy_") for name in MODULE.OVERLAY_CATALOG)
    try:
        MODULE.OverlaySpec("invalid", severe_target="policy").validate()
    except ValueError as exc:
        assert "frozen to TP6/SL4 H12" in str(exc)
    else:
        raise AssertionError("policy-net Severe target was accepted")


def test_high_rank_reliability_catalog_covers_requested_global_tails() -> None:
    expected = {
        "correctness_top05_no_k9": 0.05,
        "correctness_top10_no_k9": 0.10,
        "correctness_top15_no_k9": 0.15,
        "correctness_top20_no_k9": 0.20,
        "correctness_top25_no_k9": 0.25,
        "correctness_top30_no_k9": 0.30,
        "correctness_top35_no_k9": 0.35,
        "correctness_top40_no_k9": 0.40,
    }
    assert {
        name: MODULE.OVERLAY_CATALOG[name].correctness_training_fraction
        for name in expected
    } == expected


def test_high_rank_reliability_gate_uses_existing_causal_global_rank() -> None:
    train = pd.DataFrame({"final_score": np.arange(100, dtype=float) / 100.0})
    held = pd.DataFrame({"final_score": [0.79, 0.80, 0.84, 0.90, 0.95]})
    train_mask, held_mask, floor = MODULE._correctness_training_gate(
        train, held, retained_fraction=0.20,
    )
    assert floor == 0.80
    assert train_mask.sum() == 20
    assert held_mask.tolist() == [False, True, True, True, True]

    # Adding a future row cannot change a prior row's admission because the
    # gate never recomputes a held-window percentile.
    extended = pd.concat(
        [held, pd.DataFrame({"final_score": [1.0]})], ignore_index=True,
    )
    _, extended_mask, extended_floor = MODULE._correctness_training_gate(
        train, extended, retained_fraction=0.20,
    )
    assert extended_floor == floor
    assert extended_mask.iloc[:-1].tolist() == held_mask.tolist()


def test_reliability_context_uses_only_prior_resolved_outcomes() -> None:
    decision = pd.date_range("2025-01-01", periods=4, freq="D", tz="UTC")
    frame = pd.DataFrame(
        {
            "__decision_ts__": decision,
            "policy_label_available_ts": decision + pd.Timedelta(hours=12),
            "policy_path_valid": True,
            "policy_net_bps": [200.0, -200.0, 300.0, -300.0],
            "base_anchor_bps": 0.0,
            "base_rank": [0.8, 0.7, 0.9, 0.6],
            "consensus_rank": [0.7, 0.8, 0.6, 0.9],
            "final_score": [0.75, 0.75, 0.825, 0.675],
        }
    )
    before, groups = MODULE._causal_reliability_context(frame)
    mutated = frame.copy()
    mutated.loc[3, "policy_net_bps"] = 10_000.0
    after, _ = MODULE._causal_reliability_context(mutated)
    assert before.iloc[:4].equals(after.iloc[:4])
    assert groups["global_recent"]
    assert groups["covariance"]
    assert groups["cross_model"]


def test_risk_reliability_target_requires_risk_demote() -> None:
    MODULE.OverlaySpec(
        "ok", reliability_target="risk_residual_le_neg100",
        reliability_integration="risk_demote",
    ).validate()
    try:
        MODULE.OverlaySpec(
            "bad", reliability_target="risk_residual_le_neg100",
            reliability_integration="positive_multiplier",
        ).validate()
    except ValueError as exc:
        assert "risk targets require risk_demote" in str(exc)
    else:
        raise AssertionError("risk target accepted with positive integration")


def test_downside_and_temperature_screen_arms_are_explicitly_versioned() -> None:
    risk = MODULE.OVERLAY_CATALOG["risk_top30_residual_le_neg200_a050_no_k9"]
    assert risk.correctness_training_fraction == 0.30
    assert risk.reliability_target == "risk_residual_le_neg200"
    assert risk.reliability_integration == "risk_demote"
    assert risk.reliability_alpha == 0.50
    temperature = MODULE.OVERLAY_CATALOG[
        "correctness_top30_k9temp050_no_memberships"
    ]
    assert temperature.k9_soft_memberships is False
    assert temperature.k9_temperature_scale == 0.50


def test_global_tail_metrics_select_before_month_decomposition() -> None:
    frame = pd.DataFrame(
        {
            "final_score": np.arange(100, dtype=float),
            "policy_path_valid": True,
            "policy_net_bps": np.arange(100, dtype=float),
            "policy_gross_bps": np.arange(100, dtype=float) + 100.0,
            "__decision_ts__": pd.date_range("2025-01-01", periods=100, freq="D", tz="UTC"),
        }
    )
    global_metrics, monthly = MODULE._global_tail_metrics(frame, "unit")
    top2 = global_metrics.loc[global_metrics["tail"].eq(0.02)].iloc[0]
    assert top2["trades"] == 2
    assert monthly.loc[monthly["tail"].eq(0.02), "trades"].sum() == 2


def test_global_tail_selection_precedes_future_path_coverage_filter() -> None:
    frame = pd.DataFrame(
        {
            "final_score": np.arange(100, dtype=float),
            "policy_path_valid": [True] * 98 + [False, False],
            "policy_net_bps": np.arange(100, dtype=float),
            "policy_gross_bps": np.arange(100, dtype=float) + 100.0,
            "__decision_ts__": pd.date_range("2025-01-01", periods=100, freq="h", tz="UTC"),
        }
    )
    global_metrics, _ = MODULE._global_tail_metrics(frame, "unit")
    top2 = global_metrics.loc[global_metrics["tail"].eq(0.02)].iloc[0]
    assert top2["population_rows"] == 100
    assert top2["selected_score_rows"] == 2
    assert top2["valid_outcomes"] == 0
    assert top2["outcome_coverage"] == 0.0


def test_outcome_source_metrics_use_same_global_tail_selection() -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": [f"c{i}" for i in range(100)],
            "final_score": np.arange(100, dtype=float),
            "policy_path_valid": True,
            "policy_net_bps": np.arange(100, dtype=float),
            "policy_gross_bps": np.arange(100, dtype=float) + 100.0,
            "policy_outcome_source": ["fine"] * 99 + ["hourly"],
            "__decision_ts__": pd.date_range(
                "2025-01-01", periods=100, freq="h", tz="UTC",
            ),
        }
    )
    result = MODULE._global_tail_outcome_source_metrics(frame, "unit")
    top2 = result.loc[result["tail"].eq(0.02)].set_index("policy_outcome_source")
    assert top2["trades"].sum() == 2
    assert top2.loc["fine", "trades"] == 1
    assert top2.loc["hourly", "trades"] == 1


def test_compact_prediction_artifact_drops_raw_features_but_keeps_lineage() -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": ["a"],
            "__decision_ts__": pd.to_datetime(["2025-01-01"], utc=True),
            "__symbol__": ["BTCUSDT"],
            "final_score": [0.9],
            "policy_path_valid": [True],
            "policy_net_bps": [125.0],
            "k9_entropy": [0.2],
            "leaf_support_effective": [50.0],
            "is_admission_warmup": [True],
            "raw_model_input": [7.0],
        }
    )
    compact = MODULE._compact_prediction_artifact(frame)
    assert "raw_model_input" not in compact
    assert {
        "k9_entropy", "leaf_support_effective", "final_score",
        "is_admission_warmup",
    }.issubset(compact)


def test_score_override_replaces_only_matching_prequential_rows(tmp_path: Path) -> None:
    timestamp = pd.to_datetime(["2025-01-01", "2025-01-02"], utc=True)
    ledger = pd.DataFrame(
        {
            "candidate_id": ["a", "b"], "__decision_ts__": timestamp,
            "prequential_base_rank42": [0.1, 0.2],
            "prequential_base_anchor_bps": [1.0, 2.0],
            "prequential_consensus_rank": [0.3, 0.4],
            "prequential_upstream": [0.15, 0.25],
            "stack_is_prequential": [True, True],
        }
    )
    path = tmp_path / "override.parquet"
    pd.DataFrame(
        {
            "candidate_id": ["a", "b"], "arm": ["D1", "D1"],
            "prequential_base_rank42": [0.8, 0.9],
            "prequential_base_anchor_bps": [8.0, 9.0],
            "prequential_consensus_rank": [0.7, 0.6],
            "prequential_upstream": [0.775, 0.825],
            "stack_is_prequential": [True, True],
        }
    ).to_parquet(path, index=False)
    output, audit = MODULE._apply_score_overrides(
        ledger, path=path, arm="D1",
        evaluation_start=pd.Timestamp("2025-01-01", tz="UTC"),
        evaluation_end=pd.Timestamp("2025-02-01", tz="UTC"),
    )
    assert output["prequential_upstream"].tolist() == [0.775, 0.825]
    assert output["final_score"].tolist() == [0.775, 0.825]
    assert output["base_rank"].tolist() == [0.8, 0.9]
    assert output["base_anchor_bps"].tolist() == [8.0, 9.0]
    assert output["consensus_rank"].tolist() == [0.7, 0.6]
    assert audit["held_override_coverage"] == 1.0
