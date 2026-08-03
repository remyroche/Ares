from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.execution_entry_timing_meta import (
    ATTAINABLE_GROSS_BARRIER_RULE_ID,
    ENTRY_TIMING_SCHEMA,
    EntryAction,
    EntryTimingFeatureProvenance,
    EntryTimingTargetSpec,
    EntryTimingTrainerConfig,
    ExecutionEntryTimingBundle,
    _ConstantClassifier,
    _ConstantRegressor,
    _identity_isotonic,
    attainable_gross_barrier_action_utility,
    attainable_gross_barrier_realised,
    attainable_gross_barrier_rule_v1,
    build_counterfactual_entry_action_labels,
    load_execution_entry_timing_bundle,
    predict_execution_entry_timing_bundle,
    save_execution_entry_timing_bundle,
    train_execution_entry_timing_meta,
    validate_entry_timing_feature_contract,
    write_execution_entry_timing_artifacts,
)
from extreme_price_movements.execution_ev_labels import ExecutionLabelGeometry


def _geometry(*, sl_mult: float = 3.0) -> dict[str, object]:
    values = asdict(ExecutionLabelGeometry.from_mapping({}))
    values["sl_mult"] = sl_mult
    return values


def _target_spec(*, horizon_hours: float = 1.0, sl_mult: float = 3.0) -> EntryTimingTargetSpec:
    return EntryTimingTargetSpec(
        cost_return_col=None,
        fee_return_col="fee",
        entry_spread_bps_col="entry_spread",
        exit_spread_bps_col="exit_spread",
        horizon_hours=horizon_hours,
        long_policy_geometry=_geometry(sl_mult=sl_mult),
        short_policy_geometry=_geometry(sl_mult=sl_mult),
    )


def _path(
    start: pd.Timestamp, *, low: float = 98.0, minutes: int = 60
) -> list[dict[str, object]]:
    return [
        {
            "timestamp": start + pd.Timedelta(minutes=minute),
            "open": 100.0 - 0.2 * min(minute, 400),
            "high": 100.5 + 0.5 * minute,
            "low": low if minute == 2 else 99.0 - 0.1 * minute,
            "close": 100.0 + 0.4 * minute,
        }
        for minute in range(minutes)
    ]


def _strict_frame(
    rows: int = 4, *, with_path: bool = False, path_minutes: int = 60
) -> tuple[pd.DataFrame, dict[str, EntryTimingFeatureProvenance]]:
    times = pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC")
    values = np.linspace(-0.02, 0.03, rows)
    probability = np.full((rows, 8), 0.02)
    probability[:, 0] = 0.86
    entropy = -np.sum(probability * np.log(probability), axis=1)
    frame = pd.DataFrame(
        {
            "__ts__": times - pd.Timedelta(hours=1),
            "__decision_ts__": times,
            "execution_label_end_utc": times + pd.Timedelta(hours=12),
            "side_name": np.where(np.arange(rows) % 2, "short", "long"),
            "catboost_archetype": "trend",
            "available_at": times,
            "source_fold": "upstream-oof-0",
            "source_train_cutoff": times[0] - pd.Timedelta(minutes=1),
            "frozen_execution_ev": values,
            "frozen_ev_map": values * 0.9,
            "frozen_alpha": values * 0.5,
            "frozen_residual": values * 0.5,
            "frozen_aux_time": values,
            "frozen_aux_peak": values + 0.1,
            "frozen_aux_mae": values + 0.2,
            "frozen_aux_turn": values + 0.3,
            "frozen_aux_slope": values + 0.4,
            "frozen_entropy": entropy,
            "frozen_arch_long": (np.arange(rows) % 2 == 0).astype(float),
            "frozen_arch_short": (np.arange(rows) % 2 == 1).astype(float),
            "frozen_uncertainty": 0.1,
            "frozen_leaf_support": 2.0,
        }
    )
    for index in range(8):
        frame[f"frozen_p_{index}"] = probability[:, index]
    if with_path:
        frame["execution_future_path"] = [
            _path(time, minutes=path_minutes) for time in times
        ]
        frame["atr_1h"] = 1.0
        frame["fee"] = 0.001
        frame["entry_spread"] = 10.0
        frame["exit_spread"] = 10.0
    frozen = {
        "available_at_col": "available_at",
        "frozen_bundle_id": "execution-ev-final-v1",
    }
    oof = {
        **frozen,
        "oof_fold_col": "source_fold",
        "source_train_cutoff_col": "source_train_cutoff",
    }
    provenance = {
        "frozen_execution_ev": EntryTimingFeatureProvenance("execution_ev_prediction", "execution EV OOF", cost_spread_aware=True, **oof),
        "frozen_ev_map": EntryTimingFeatureProvenance("execution_ev_mapping", "execution EV map OOF", **oof),
        "frozen_alpha": EntryTimingFeatureProvenance("alpha_outputs", "alpha OOF", **oof),
        "frozen_residual": EntryTimingFeatureProvenance("residual_outputs", "residual OOF", **oof),
        "frozen_aux_time": EntryTimingFeatureProvenance("auxiliary_heads", "time auxiliary OOF", **oof),
        "frozen_aux_peak": EntryTimingFeatureProvenance("auxiliary_heads", "peak auxiliary OOF", **oof),
        "frozen_aux_mae": EntryTimingFeatureProvenance("auxiliary_heads", "MAE auxiliary OOF", **oof),
        "frozen_aux_turn": EntryTimingFeatureProvenance("auxiliary_heads", "turn auxiliary OOF", **oof),
        "frozen_aux_slope": EntryTimingFeatureProvenance("auxiliary_heads", "slope auxiliary OOF", **oof),
        "frozen_entropy": EntryTimingFeatureProvenance("catboost_entropy", "CatBoost entropy OOF", **oof),
        "frozen_arch_long": EntryTimingFeatureProvenance("side_archetypes", "frozen base archetype", **frozen),
        "frozen_arch_short": EntryTimingFeatureProvenance("side_archetypes", "frozen base archetype", **frozen),
        "frozen_uncertainty": EntryTimingFeatureProvenance("uncertainty", "frozen uncertainty", **frozen),
        "frozen_leaf_support": EntryTimingFeatureProvenance("leaf_support", "frozen leaf support", **frozen),
    }
    provenance.update({
        f"frozen_p_{index}": EntryTimingFeatureProvenance("catboost_probabilities", "CatBoost probability OOF", **oof)
        for index in range(8)
    })
    return frame, provenance


def test_counterfactual_labels_use_exact_60_bar_geometry_aware_policy_and_costs_once() -> None:
    frame, _ = _strict_frame(1, with_path=True)
    # The passive order is live from the first executable minute. Keep that
    # first bar above the limit so the deliberate minute-two adverse touch
    # exercises the ambiguous-fill-bar exclusion below.
    frame.at[0, "execution_future_path"][0]["low"] = 99.9
    actions = (
        EntryAction("enter_now"),
        EntryAction("adverse_limit", wait_minutes=3, adverse_offset_atr=0.5),
        EntryAction("wait_market", wait_minutes=75),
    )
    labels = build_counterfactual_entry_action_labels(
        frame,
        action_grid=actions,
        target_spec=_target_spec(),
    )
    repeat = build_counterfactual_entry_action_labels(
        frame,
        action_grid=actions,
        target_spec=_target_spec(),
    )
    pd.testing.assert_frame_equal(labels, repeat)
    limit = labels.loc[labels["action_kind"].eq("adverse_limit")].iloc[0]
    no_fill = labels.loc[labels["action_kind"].eq("wait_market")].iloc[0]
    assert limit["fill_indicator"] == 1.0
    # The no-decision-price contract anchors the passive limit at the signed
    # first executable 1m open at __decision_ts__ (100.0).
    assert limit["raw_fill_price"] == pytest.approx(99.5)
    assert limit["fill_price"] == pytest.approx(99.5 * 1.001)
    assert limit["cost_accounting_mode"] == "fee_once_entry_spread_once_exit_spread_once"
    assert np.isfinite(limit["conditional_post_fill_executable_ev"])
    assert limit["fill_bar_intrabar_ambiguity"]
    assert limit["policy_simulation_start_utc"] == frame.loc[0, "__decision_ts__"] + pd.Timedelta(minutes=2)
    now = labels.loc[labels["action_kind"].eq("enter_now")].iloc[0]
    assert now["counterfactual_label_end_utc"] == frame.loc[0, "__decision_ts__"] + pd.Timedelta(hours=1)
    assert now["execution_exit_reason"] in {"timeout", "trailing"}
    assert no_fill["no_fill_indicator"] == 1.0
    assert no_fill["missed_opportunity_loss"] > 0.0


def test_named_attainable_gross_barrier_rule_uses_executable_gross_not_mfe() -> None:
    action = attainable_gross_barrier_rule_v1()
    assert ATTAINABLE_GROSS_BARRIER_RULE_ID == "AGBR-L60-K0.25-v1"
    assert action.action_id == "adverse_limit_60m_0.2500atr"
    assert action.kind == "adverse_limit"
    assert action.wait_minutes == 60
    assert action.adverse_offset_atr == pytest.approx(0.25)

    # A hypothetical large MFE is intentionally irrelevant: a gross result
    # below the once-only fee does not count as attainable realization.
    assert not attainable_gross_barrier_realised(
        filled=True,
        post_fill_gross_ev=0.009,
        fee_return=0.010,
    )
    assert attainable_gross_barrier_realised(
        filled=True,
        post_fill_gross_ev=0.012,
        fee_return=0.010,
        target_net_buffer=0.002,
    )
    assert not attainable_gross_barrier_realised(
        filled=False,
        post_fill_gross_ev=0.100,
        fee_return=0.010,
    )


def test_named_attainable_gross_barrier_rule_penalizes_missed_opportunity() -> None:
    assert attainable_gross_barrier_action_utility(
        filled=False,
        post_fill_net_ev=np.nan,
        enter_now_net_ev=0.025,
    ) == pytest.approx(-0.025)
    assert attainable_gross_barrier_action_utility(
        filled=False,
        post_fill_net_ev=np.nan,
        enter_now_net_ev=-0.025,
    ) == pytest.approx(0.0)
    assert attainable_gross_barrier_action_utility(
        filled=True,
        post_fill_net_ev=0.012,
        enter_now_net_ev=0.025,
    ) == pytest.approx(0.012)


def test_counterfactual_labels_use_canonical_geometry_stop_not_terminal_close() -> None:
    frame, _ = _strict_frame(1, with_path=True)
    labels = build_counterfactual_entry_action_labels(
        frame,
        action_grid=(EntryAction("enter_now"),),
        target_spec=_target_spec(sl_mult=0.5),
    )
    row = labels.iloc[0]
    assert row["execution_exit_reason"] == "full_stop"
    assert row["execution_exit_bar"] == 0
    assert row["post_fill_net_ev"] < 0.0


def test_counterfactual_labels_reject_nonexact_1m_horizon_and_terminal_timestamp() -> None:
    frame, _ = _strict_frame(1, with_path=True)
    path = _path(frame.loc[0, "__decision_ts__"], minutes=59)
    frame.at[0, "execution_future_path"] = path
    with pytest.raises(ValueError, match="exact fixed 1m horizon length"):
        build_counterfactual_entry_action_labels(
            frame,
            target_spec=_target_spec(),
        )
    path = _path(frame.loc[0, "__decision_ts__"])
    path[-1]["timestamp"] = frame.loc[0, "__decision_ts__"] + pd.Timedelta(minutes=61)
    frame.at[0, "execution_future_path"] = path
    with pytest.raises(ValueError, match="end exactly"):
        build_counterfactual_entry_action_labels(frame, target_spec=_target_spec())


def test_counterfactual_labels_reject_all_in_action_cost_without_explicit_opt_in() -> None:
    frame, _ = _strict_frame(1, with_path=True)
    frame["all_in_cost"] = 0.002
    with pytest.raises(ValueError, match="all-in execution cost is rejected"):
        build_counterfactual_entry_action_labels(
            frame,
            target_spec=EntryTimingTargetSpec(
                cost_return_col="all_in_cost",
                horizon_hours=1.0,
                long_policy_geometry=_geometry(),
                short_policy_geometry=_geometry(),
            ),
        )


def test_feature_contract_requires_protected_cost_aware_ev_and_provenance() -> None:
    frame, provenance = _strict_frame()
    names, execution_ev = validate_entry_timing_feature_contract(frame, provenance)
    assert "frozen_execution_ev" in names
    assert execution_ev == "frozen_execution_ev"
    bad = dict(provenance)
    bad["frozen_execution_ev"] = EntryTimingFeatureProvenance(
        "execution_ev_prediction", "in-sample execution EV", available_at_col="available_at", frozen_bundle_id=""
    )
    with pytest.raises(ValueError, match="in-sample|OOF"):
        validate_entry_timing_feature_contract(frame, bad)
    leaked = frame.copy()
    leaked["source_train_cutoff"] = leaked["__decision_ts__"]
    with pytest.raises(ValueError, match="strictly before"):
        validate_entry_timing_feature_contract(leaked, provenance)
    with pytest.raises(ValueError, match="__decision_ts__"):
        EntryTimingTrainerConfig(decision_time_col="__ts__")


def test_bundle_roundtrip_scores_without_train_only_path_fields(tmp_path) -> None:
    frame, provenance = _strict_frame(2)
    config = EntryTimingTrainerConfig(
        strict_feature_families=False,
        action_grid=(EntryAction("enter_now"),),
        min_train_rows=4,
    )
    feature_names = ("frozen_execution_ev",)
    state = {
        "lgbm_fill": _ConstantClassifier(1.0),
        "lgbm_adverse": _ConstantClassifier(1.0),
        "lgbm_delta": _ConstantRegressor(0.0),
        "ridge_fill": _ConstantClassifier(1.0),
        "ridge_adverse": _ConstantClassifier(0.0),
        "ridge_delta": _ConstantRegressor(0.0),
        "fixed_grid": {"fill_probability": 1.0, "adverse_probability": 0.0, "filled_delta_ev": 0.0},
        "isotonic": {"lgbm": _identity_isotonic(), "ridge_logistic": _identity_isotonic(), "fixed_grid": _identity_isotonic()},
    }
    bundle = ExecutionEntryTimingBundle(
        schema=ENTRY_TIMING_SCHEMA,
        config={**asdict(config), "action_grid": [asdict(EntryAction("enter_now"))]},
        target_spec=EntryTimingTargetSpec(),
        provenance={"frozen_execution_ev": provenance["frozen_execution_ev"]},
        feature_names=feature_names,
        execution_ev_feature="frozen_execution_ev",
        decision_policy={"missed_opportunity_penalty": 1.0, "adverse_first_penalty": 0.004, "max_wait_minutes": 0, "max_adverse_offset_atr": 0.0},
        models={"long": {"enter_now": state}, "short": {"enter_now": state}},
        report={}, input_fingerprint="input", bundle_fingerprint="bundle",
        oof_action_predictions=pd.DataFrame(), oof_recommendations=pd.DataFrame(), oof_provenance=pd.DataFrame(),
    )
    path = save_execution_entry_timing_bundle(bundle, tmp_path / "bundle.joblib")
    loaded = load_execution_entry_timing_bundle(path)
    scored = predict_execution_entry_timing_bundle(loaded, frame.loc[:, ["__decision_ts__", "side_name", "catboost_archetype", "available_at", "frozen_execution_ev"]])
    assert scored["recommended_action_id"].eq("enter_now").all()
    np.testing.assert_allclose(
        scored["expected_action_ev"].to_numpy(),
        frame["frozen_execution_ev"].to_numpy(),
    )
    assert np.isfinite(scored["expected_action_ev"]).all()


def test_training_metrics_are_outer_oof_and_final_refit_is_inference_only(tmp_path) -> None:
    pytest.importorskip("lightgbm")
    frame, provenance = _strict_frame(48, with_path=True)
    config = EntryTimingTrainerConfig(
        n_splits=2,
        min_train_rows=8,
        purge_hours=1.0,
        embargo_hours=0.0,
        n_estimators=8,
        early_stopping_rounds=4,
        hpo_trials=0,
        decision_hpo_trials=0,
        action_grid=(
            EntryAction("enter_now"),
            EntryAction("wait_market", wait_minutes=2),
            EntryAction("adverse_limit", wait_minutes=2, adverse_offset_atr=0.5),
        ),
    )
    target_spec = _target_spec()
    bundle = train_execution_entry_timing_meta(
        frame, provenance, config=config, target_spec=target_spec
    )
    scored = bundle.oof_provenance.loc[
        bundle.oof_provenance["entry_timing_oof_fold"].notna()
    ]
    assert not scored.empty
    assert (
        scored["entry_timing_oof_train_decision_cutoff_utc"]
        < scored["entry_timing_oof_validation_start_utc"]
    ).all()
    overall = bundle.report["diagnostics"].query("arm == 'lgbm' and scope == 'overall'").iloc[0]
    assert overall["rows"] == len(scored)
    assert "final refit is excluded" in bundle.report["oof_contract"]
    paths = write_execution_entry_timing_artifacts(bundle, tmp_path / "artifacts")
    assert {"bundle", "diagnostics", "oof_actions", "oof_recommendations", "provenance_manifest", "inference_manifest", "report"}.issubset(paths)
    assert all(path.is_file() for path in paths.values())
    scoring = frame.drop(columns=["execution_future_path", "atr_1h", "fee", "entry_spread", "exit_spread"])
    with pytest.raises(ValueError, match="realised|realized"):
        predict_execution_entry_timing_bundle(
            bundle, scoring.assign(execution_net_ev_12h=0.01)
        )


def test_training_accepts_causal_rolling_upstream_oof_cutoffs() -> None:
    frame, provenance = _strict_frame(24, with_path=True)
    frame["source_train_cutoff"] = frame["__decision_ts__"] - pd.Timedelta(minutes=1)
    config = EntryTimingTrainerConfig(
        n_splits=2,
        min_train_rows=6,
        purge_hours=1.0,
        embargo_hours=0.0,
        hpo_trials=0,
        decision_hpo_trials=0,
        strict_feature_families=False,
        action_grid=(EntryAction("enter_now"),),
    )
    bundle = train_execution_entry_timing_meta(
        frame, provenance, config=config, target_spec=_target_spec()
    )
    assert bundle.report["oof_contract"]
