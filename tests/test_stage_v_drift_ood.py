import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.stage_v_drift_ood import (
    STAGE_V_FEATURE_COLUMNS,
    StageVContract,
    attach_stage_v_context,
    fit_stage_v_drift_ood_state,
    prequential_stage_v_drift_ood_features,
    resolve_stage_v_mda_groups,
    transform_stage_v_drift_ood_features,
)


def _mda_audit() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "group_id": ["group_00", "group_01", "screen_ignored"],
            "group_kind": ["correlation", "correlation", "screen_correlation"],
            "features": ["a|b", "c|d", "a|d"],
            "group_mda_lower_95": [0.12, 0.04, 0.50],
            "group_mda_mean": [0.14, 0.05, 0.60],
        }
    )


def _reference(seed: int = 12, n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    return pd.DataFrame(
        {
            "a": a,
            "b": a + rng.normal(scale=0.1, size=n),
            "c": rng.normal(size=n),
            "d": rng.normal(size=n),
            "unused": 1.0,
        }
    )


def test_groups_use_positive_correlation_mda_groups_only():
    groups = resolve_stage_v_mda_groups(_mda_audit(), available_columns=["a", "b", "c", "d"])
    assert [g["group_id"] for g in groups] == ["group_00", "group_01"]
    assert groups[0]["members"] == ["a", "b"]


def test_train_only_state_is_side_layer_scoped_and_soft_context_is_finite():
    ref = _reference()
    state = fit_stage_v_drift_ood_state(
        ref,
        contract=StageVContract("long", "meta"),
        mda_audit=_mda_audit(),
    )
    assert state["enabled"] is True
    assert state["reference_role"] == "train_only"
    assert state["soft_context_only"] is True
    out = transform_stage_v_drift_ood_features(ref.iloc[:10], state, contract=StageVContract("long", "meta"))
    assert list(out.columns) == list(STAGE_V_FEATURE_COLUMNS)
    assert np.isfinite(out.to_numpy(dtype=np.float32)).all()
    assert out["stage_v_reference_ready"].eq(1.0).all()
    assert out["stage_v_ood_score"].between(0.0, 1.0).all()
    with pytest.raises(ValueError, match="side/layer"):
        transform_stage_v_drift_ood_features(ref.iloc[:2], state, contract=StageVContract("short", "meta"))


def test_training_only_coactivation_merges_mda_groups_that_fire_together():
    ref = _reference()
    ref["c"] = ref["a"] + 0.01
    ref["d"] = ref["b"] - 0.01
    state = fit_stage_v_drift_ood_state(ref, contract=StageVContract("long", "base"), mda_audit=_mda_audit())
    assert state["coactivation_fit"]["source"] == "training_only_mda_group_activation"
    assert len(state["groups"]) == 1
    assert set(state["groups"][0]["source_mda_group_ids"]) == {"group_00", "group_01"}


def test_shifted_joint_activation_increases_ood_without_batch_dependence():
    ref = _reference()
    state = fit_stage_v_drift_ood_state(ref, contract=StageVContract("long", "base"), mda_audit=_mda_audit())
    normal = transform_stage_v_drift_ood_features(ref.iloc[:4], state, contract=StageVContract("long", "base"))
    shifted_rows = ref.iloc[:4].copy()
    shifted_rows.loc[:, ["a", "b"]] += 6.0
    shifted = transform_stage_v_drift_ood_features(shifted_rows, state, contract=StageVContract("long", "base"))
    assert shifted["stage_v_group_coactivation_max"].mean() > normal["stage_v_group_coactivation_max"].mean()
    assert shifted["stage_v_ood_score"].mean() > normal["stage_v_ood_score"].mean()
    single = transform_stage_v_drift_ood_features(shifted_rows.iloc[:1], state, contract=StageVContract("long", "base"))
    assert np.allclose(single.iloc[0], shifted.iloc[0], rtol=0.0, atol=1e-7)


def test_prequential_never_uses_current_or_future_timestamp_rows():
    frame = _reference(n=96)
    ts = pd.date_range("2026-01-01", periods=48, freq="h", tz="UTC").repeat(2)
    out, audit = prequential_stage_v_drift_ood_features(
        frame,
        timestamps=ts,
        contract=StageVContract("short", "meta"),
        mda_audit=_mda_audit(),
        min_reference_rows=8,
        refresh_every_timestamps=1,
    )
    assert list(out.columns) == list(STAGE_V_FEATURE_COLUMNS)
    assert audit["strictly_prior_reference"].all()
    assert (audit["reference_rows"].iloc[:4] == [0, 2, 4, 6]).all()
    assert out.iloc[:8]["stage_v_reference_ready"].eq(0.0).all()
    # Changing only later rows must not alter earlier prequential features.
    changed = frame.copy()
    changed.loc[32:, ["a", "b", "c", "d"]] += 100.0
    changed_out, _ = prequential_stage_v_drift_ood_features(
        changed,
        timestamps=ts,
        contract=StageVContract("short", "meta"),
        mda_audit=_mda_audit(),
        min_reference_rows=8,
        refresh_every_timestamps=1,
    )
    assert np.allclose(out.iloc[:32], changed_out.iloc[:32], rtol=0.0, atol=1e-7)


def test_context_attachment_preserves_pooled_global_score_order_unchanged():
    ref = _reference()
    state = fit_stage_v_drift_ood_state(ref, contract=StageVContract("long", "base"), mda_audit=_mda_audit())
    context = transform_stage_v_drift_ood_features(ref.iloc[:4], state, contract=StageVContract("long", "base"))
    ledger = pd.DataFrame({"candidate_id": ["a", "b", "c", "d"], "score_bps": [3.0, 1.0, 4.0, 2.0]})
    attached = attach_stage_v_context(ledger, context, candidate_ids=ledger["candidate_id"])
    assert attached["score_bps"].tolist() == ledger["score_bps"].tolist()
    assert attached.sort_values("score_bps", ascending=False)["candidate_id"].tolist() == ["c", "a", "d", "b"]
