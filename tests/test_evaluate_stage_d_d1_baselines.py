from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_stage_d_d1_baselines import attach_causal_buckets, daily_comparisons, evaluate, run


def _counterfactuals() -> pd.DataFrame:
    decision = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    continue_net = np.array([90.0, 40.0, -20.0, 15.0])
    exit_net = np.array([110.0, 10.0, -10.0, 15.0])
    cost = np.full(4, 10.0)
    return pd.DataFrame({
        "candidate_id": ["c1", "c2", "c3", "c4"], "side": ["long", "short", "long", "short"],
        "action_decision_ts": decision, "first_clear_bar_index": [0, 5, 30, 300],
        "net_continue_gross_bps": continue_net + cost, "net_continue_cost_bps": cost, "net_continue_bps": continue_net,
        "net_exit_now_gross_bps": exit_net + cost, "net_exit_now_cost_bps": cost, "net_exit_now_bps": exit_net,
        "delta_continue_bps": continue_net - exit_net,
    })


def _features() -> pd.DataFrame:
    decision = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    return pd.DataFrame({"candidate_id": ["c1", "c2", "c3", "c4"], "action_decision_ts": decision, "feature_available_ts": decision, "volume_z_at_clear": [-2.0, -.5, .5, 2.0], "realised_volatility": [10.0, 30.0, 70.0, 150.0]})


def test_d1_uses_fixed_paired_candidate_population_and_causal_buckets() -> None:
    rows, status = attach_causal_buckets(_counterfactuals(), _features())
    assert rows.candidate_id.tolist() == ["c1", "c2", "c3", "c4"]
    assert rows.time_to_clear_bucket.tolist() == ["01-05m", "06-15m", "31-60m", "241-480m"]
    assert rows.volume_bucket.tolist() == ["z<=-1", "-1<z<=0", "0<z<=1", "z>1"]
    assert rows.volatility_bucket.tolist() == ["<=25bps", "25-50bps", "50-100bps", ">100bps"]
    assert status["regime_bucket"] == "NOT_REPORTED_A8_REJECTED_OOF_LINEAGE"


def test_d1_reports_correct_paired_economics_and_giveback() -> None:
    rows, _ = attach_causal_buckets(_counterfactuals(), _features())
    summary, paired, daily, facts = evaluate(rows)
    overall = summary.loc[summary.group_type.eq("overall")].iloc[0]
    assert overall.continue_net_mean_bps == pytest.approx(31.25)
    assert overall.exit_net_mean_bps == pytest.approx(31.25)
    assert overall.exit_minus_continue_mean_bps == pytest.approx(0.0)
    assert overall.exit_better_row_rate == pytest.approx(.5)
    assert overall.loss_avoided_sum_bps == pytest.approx(30.0)
    assert overall.loss_avoided_mean_bps == pytest.approx(7.5)
    assert overall.false_exit_opportunity_cost_sum_bps == pytest.approx(30.0)
    assert overall.false_exit_opportunity_cost_mean_bps == pytest.approx(7.5)
    assert paired.loc[paired.candidate_id.eq("c1"), "mechanical_exit_is_better"].item()
    assert len(daily) == 4
    assert facts["candidate_ids_fixed_across_b0_b1"]
    assert facts["baseline_uplift_exit_minus_continue_mean_bps"] == pytest.approx(0.0)


def test_day_block_bootstrap_recomputes_pooled_effect_with_unequal_day_support() -> None:
    # Day one has 100 rows at +10 bps; day two has one row at -100 bps.
    # Equal averaging of the two day means is -45 bps, while the required
    # pooled estimator is +900 / 101 bps.
    rows = pd.DataFrame({
        "candidate_id": [f"large_{i}" for i in range(100)] + ["small"],
        "utc_day": ["2024-01-01"] * 100 + ["2024-01-02"],
        "net_continue_bps": [0.0] * 101,
        "net_exit_now_bps": [10.0] * 100 + [-100.0],
    })
    daily, report = daily_comparisons(rows)
    assert daily.exit_minus_continue_mean_bps.mean() == pytest.approx(-45.0)
    assert report["exit_minus_continue_pooled_mean_bps"] == pytest.approx(900.0 / 101.0)
    assert report["bootstrap_estimator"] == "resample whole UTC-day blocks; sum sampled EXIT_MINUS_CONTINUE bps / sum sampled rows"


def test_d1_rejects_feature_pack_with_missing_candidate_ids() -> None:
    with pytest.raises(ValueError, match="does not cover"):
        attach_causal_buckets(_counterfactuals(), _features().iloc[:-1])


def test_d1_allows_explicit_absence_of_exact_volume_but_requires_volatility() -> None:
    rows, status = attach_causal_buckets(_counterfactuals(), _features().drop(columns="volume_z_at_clear"))
    assert "volume_bucket" not in rows
    assert status["volume_bucket"] == "NOT_REPORTED_A3_SOURCE_FIELD_ABSENT_OR_UNAVAILABLE"
    with pytest.raises(ValueError, match="realised_volatility"):
        attach_causal_buckets(_counterfactuals(), _features().drop(columns="realised_volatility"))


def test_d1_seals_deterministic_outputs(tmp_path: Path) -> None:
    counter, feature, output = tmp_path / "counter.parquet", tmp_path / "features.parquet", tmp_path / "out"
    _counterfactuals().to_parquet(counter, index=False)
    _features().to_parquet(feature, index=False)
    manifest = run(counterfactuals_path=counter, features_path=feature, output=output)
    assert manifest["status"] == "SEALED_DETERMINISTIC_D1_BASELINES_NO_MODEL_OR_POLICY_CHANGE"
    persisted = json.loads((output / "manifest.json").read_text())
    for name, expected in persisted["outputs_sha256"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected
    assert (output / "stage_d_d1_paired_utc_day_comparisons.parquet").exists()
