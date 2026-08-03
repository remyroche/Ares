from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.run_stage_d_compact_action_model import (
    D2, MARGINS, apply_preprocess, choose_margin, compact_readmission_decision,
    fit_preprocess, replay, require_d2, result_table, train_mask,
)


def _predictions() -> pd.DataFrame:
    return pd.DataFrame({
        "candidate_id": ["a", "b", "c"], "source_symbol": ["X"]*3, "side": ["long"]*3,
        "time_to_clear_bucket": ["01-15m"]*3, "volatility_bucket": ["<=25bps"]*3,
        "action_decision_ts": pd.date_range("2024-04-01", periods=3, freq="D", tz="UTC"),
        "predicted_delta_continue_bps": [-1., 30., 60.], "predicted_continue_probability": [.2,.6,.8],
        "delta_continue_bps": [-10., 20., 50.], "continue_better": [0,1,1],
        "net_continue_gross_bps": [100.,100.,100.], "net_continue_cost_bps": [10.,10.,10.], "net_continue_bps": [90.,90.,90.],
        "net_exit_now_gross_bps": [110.,80.,50.], "net_exit_now_cost_bps": [10.,10.,10.], "net_exit_now_bps": [100.,70.,40.],
        "split": ["development_oof"]*3, "raw_predicted_delta_bps": [-1.,30.,60.], "arm": ["compact"]*3,
    })


def test_action_threshold_uses_development_data_only() -> None:
    winner,evidence=choose_margin(_predictions())
    assert winner in MARGINS
    assert set(evidence.margin_bps)==set(MARGINS)
    assert _predictions().split.eq("development_oof").all()


def test_threshold_is_absolute_bps_and_never_top_k() -> None:
    z=replay(_predictions(),25.)
    assert z.action.tolist()==["EXIT_NOW","CONTINUE_FROZEN_POLICY","CONTINUE_FROZEN_POLICY"]
    changed=_predictions().iloc[::-1].reset_index(drop=True)
    assert replay(changed,25.).set_index("candidate_id").action.to_dict()==z.set_index("candidate_id").action.to_dict()


def test_no_entry_or_portfolio_policy_is_changed() -> None:
    source=Path(__file__).resolve().parents[1].joinpath("scripts/run_stage_d_compact_action_model.py").read_text()
    assert "top-k" in source and "NO_ENTRY_OR_PORTFOLIO_POLICY_CHANGE" in source
    assert all(token not in source for token in ("simple_policy_optimiser(", "portfolio_limit=", "concurrency_limit=", "position_size="))


def test_final_preprocessing_is_training_only_and_feature_count_controlled() -> None:
    train=pd.DataFrame({"a":np.arange(200,dtype=float),"b":np.sin(np.arange(200))})
    state=fit_preprocess(train,["a","b"],np.arange(200,dtype=float),7,cap=1)
    assert len(state["selected"])==1 and state["feature_cap"]==1
    first=apply_preprocess(pd.DataFrame({"a":[1e9],"b":[1e9]}),state)
    second=apply_preprocess(pd.DataFrame({"a":[2e9],"b":[2e9]}),state)
    pd.testing.assert_frame_equal(first,second)  # both use frozen training clips


def test_final_fold_uses_resolved_labels_only() -> None:
    start=pd.Timestamp("2024-08-01T00:00Z")
    f=pd.DataFrame({"action_decision_ts":[start-pd.Timedelta(hours=13),start-pd.Timedelta(hours=11)],"label_available_ts":[start-pd.Timedelta(minutes=1)]*2})
    assert train_mask(f,start).tolist()==[True,False]


def test_runner_waits_for_canonical_d2_v4() -> None:
    if not D2.exists():
        with pytest.raises(FileNotFoundError,match="canonical D2 v4"):
            require_d2()


def test_compact_readmission_is_development_metric_only() -> None:
    full={"net_policy_bps":94.,"mae_bps":135.,"spearman_ic":.79,"calibration_slope":1.,"calibration_intercept_bps":2.}
    a0={"net_policy_bps":95.,"mae_bps":134.,"spearman_ic":.80,"calibration_slope":1.,"calibration_intercept_bps":2.}
    assert compact_readmission_decision(full,a0)
    assert not compact_readmission_decision(full,{**a0,"mae_bps":136.})


def test_result_table_preserves_arm_and_symbol_slices() -> None:
    table=result_table(replay(_predictions(),25.))
    assert table.arm.unique().tolist()==["compact"]
    assert ((table.dimension=="symbol") & (table.value=="X")).any()


def test_lineage_gate_is_evidence_derived_and_sources_are_archived_by_hash() -> None:
    source=Path(__file__).resolve().parents[1].joinpath("scripts/run_stage_d_compact_action_model.py").read_text()
    assert "bool(lineage['passed'])" in source
    assert "'runner_sha256':sha(Path(__file__))" in source
    assert "'tests_sha256':sha(ROOT/'tests/test_stage_d_compact_action_model.py')" in source


def test_lightgbm_is_forced_to_single_thread_deterministic_mode() -> None:
    source=Path(__file__).resolve().parents[1].joinpath("scripts/run_stage_d_compact_action_model.py").read_text()
    assert source.count("n_jobs=1") == 2
    assert source.count("deterministic=True") == 2
    assert source.count("force_col_wise=True") == 2
