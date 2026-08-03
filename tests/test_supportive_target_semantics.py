"""Tests for censored, conditional path-supportive target semantics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.supportive_target_semantics import (
    SupportiveTargetContractError,
    materialize_supportive_target_semantics,
)
from scripts.materialize_supportive_target_semantics import run


def _labels() -> pd.DataFrame:
    decision = pd.date_range("2026-01-01", periods=3, freq="D", tz="UTC")
    return pd.DataFrame({
        "candidate_id": ["a", "b", "c"],
        "symbol": ["AAA/USD:USD"] * 3,
        "side": ["long", "short", "long"],
        "decision_ts": decision,
        "label_end_ts": decision + pd.Timedelta(hours=12),
        "label_available_ts": decision + pd.Timedelta(hours=12),
        "__path_auxiliary_target_valid__": [1, 1, 0],
        "__time_to_first_meaningful_mfe_target_valid__": [1, 1, 1],
        "__meaningful_mfe_reached_12h__": [1, 0, 1],
        "__peak_mfe_atr_12h__": [2.0, 0.0, 3.0],
        "__mae_before_meaningful_mfe_atr_12h__": [0.5, 1.0, 0.7],
        "__time_to_first_meaningful_mfe_hours_12h__": [3.0, 12.0, 2.0],
        "clean_economic_favorable_first": [1, 0, 1],
        "adverse_first": [0, 1, 0],
        "same_minute_favorable_adverse_conflict": [0, 0, 0],
        "first_favorable_minute": [30.0, np.nan, 15.0],
        "__mfe_persistence_path_efficiency_12h__": [0.7, 0.3, 0.9],
        "__adverse_trough_atr_12h__": [0.0, 1.2, 0.5],
        "__adverse_trough_recovery_50pct_confirmed_2bars_12h__": [0, 1, 1],
    })


def test_conditional_peak_and_mae_are_never_zero_filled_for_unreached_or_invalid_rows() -> None:
    labels, contract = materialize_supportive_target_semantics(_labels())
    assert labels.target_peak_mfe_atr_given_meaningful_mfe_valid.tolist() == [1, 0, 0]
    assert labels.target_mae_before_meaningful_mfe_atr_given_meaningful_mfe_valid.tolist() == [1, 0, 0]
    assert labels.target_peak_mfe_atr_given_meaningful_mfe.tolist()[0] == pytest.approx(2.0)
    assert labels.target_mae_before_meaningful_mfe_atr_given_meaningful_mfe.tolist()[0] == pytest.approx(0.5)
    assert np.isnan(labels.target_peak_mfe_atr_given_meaningful_mfe.iloc[1:]).all()
    assert np.isnan(labels.target_mae_before_meaningful_mfe_atr_given_meaningful_mfe.iloc[1:]).all()
    assert contract["model_input_eligible"] is False


def test_time_targets_are_right_censored_with_hazards_only_while_at_risk() -> None:
    labels, _ = materialize_supportive_target_semantics(_labels())
    # Row a reaches meaningful MFE at 3h: it is at risk for 0-1, 1-2 and 2-4,
    # reaches in 2-4, and is not assigned a post-event 4-8 zero.
    assert labels.loc[0, "target_meaningful_mfe_observed_time_hours"] == pytest.approx(3.0)
    assert labels.loc[0, "target_meaningful_mfe_hazard_0_1h"] == 0.0
    assert labels.loc[0, "target_meaningful_mfe_hazard_1_2h"] == 0.0
    assert labels.loc[0, "target_meaningful_mfe_hazard_2_4h"] == 1.0
    assert labels.loc[0, "target_meaningful_mfe_hazard_4_8h_valid"] == 0
    assert np.isnan(labels.loc[0, "target_meaningful_mfe_hazard_4_8h"])
    assert labels.loc[0, "target_meaningful_mfe_cumulative_reach_by_2h"] == 0
    assert labels.loc[0, "target_meaningful_mfe_cumulative_reach_by_4h"] == 1

    # Row b is censored, not observed at a synthetic 12h event time.
    assert labels.loc[1, "target_meaningful_mfe_event_observed"] == 0
    assert np.isnan(labels.loc[1, "target_meaningful_mfe_observed_time_hours"])
    assert labels.loc[1, "target_meaningful_mfe_censor_time_hours"] == pytest.approx(12.0)
    assert labels.loc[1, "target_meaningful_mfe_hazard_8_12h_valid"] == 1
    assert labels.loc[1, "target_meaningful_mfe_hazard_8_12h"] == 0.0

    # The opportunity event uses its own first-favourable time and is likewise
    # censored for an adverse/timeout row.
    assert labels.loc[0, "target_opportunity_observed_time_hours"] == pytest.approx(0.5)
    assert labels.loc[0, "target_opportunity_hazard_0_1h"] == 1.0
    assert labels.loc[1, "target_opportunity_event_observed"] == 0
    assert np.isnan(labels.loc[1, "target_opportunity_observed_time_hours"])


def test_opportunity_adverse_persistence_and_recovery_keep_their_validity_domains() -> None:
    labels, _ = materialize_supportive_target_semantics(_labels())
    assert labels.support_opportunity.tolist()[:2] == [1.0, 0.0]
    assert labels.support_adverse.tolist()[:2] == [0.0, 1.0]
    assert labels.support_persistence_given_meaningful_mfe_valid.tolist() == [1, 0, 0]
    assert labels.support_persistence_given_meaningful_mfe.iloc[0] == pytest.approx(0.7)
    assert np.isnan(labels.support_persistence_given_meaningful_mfe.iloc[1])
    assert labels.support_adverse_recovery_50pct_confirmed_valid.tolist() == [0, 1, 0]
    assert np.isnan(labels.support_adverse_recovery_50pct_confirmed.iloc[0])
    assert labels.support_adverse_recovery_50pct_confirmed.iloc[1] == 1.0


def test_source_horizon_and_observed_event_time_contract_fail_closed() -> None:
    bad_horizon = _labels()
    bad_horizon.loc[0, "label_available_ts"] += pd.Timedelta(minutes=1)
    with pytest.raises(SupportiveTargetContractError, match="availability"):
        materialize_supportive_target_semantics(bad_horizon)
    bad_time = _labels()
    bad_time.loc[0, "__time_to_first_meaningful_mfe_hours_12h__"] = 13.0
    with pytest.raises(SupportiveTargetContractError, match="within the horizon"):
        materialize_supportive_target_semantics(bad_time)


def test_sidecar_runner_does_not_modify_the_source_pack(tmp_path: Path) -> None:
    source = tmp_path / "supportive_labels.parquet"
    _labels().to_parquet(source, index=False)
    source_hash_before = source.read_bytes()
    manifest = run(source=source, output=tmp_path / "sidecar")
    assert manifest["status"] == "MATERIALIZED_RESEARCH_LABEL_SIDECAR_ONLY"
    assert source.read_bytes() == source_hash_before
    assert (tmp_path / "sidecar/supportive_target_semantics.parquet").is_file()
    contract = json.loads((tmp_path / "sidecar/supportive_target_semantics_contract.json").read_text())
    assert contract["prohibition"].startswith("all output columns")
