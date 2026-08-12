from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.stage_ii_direct_fq3_bridge import (
    StageIIDirectFQ3Candidate,
    StageIIDirectFQ3Error,
    StageIIDirectFQ3Spec,
    materialize_stage_ii_direct_fq3_handoff,
    run_stage_ii_direct_fq3_archetype_funnel,
    score_frozen_stage_ii_direct_fq3,
    validate_stage_ii_direct_fq3_ledger,
)
from extreme_price_movements.stage_ii_meta_archetypes import StageIIMetaArchetypeConfig
from extreme_price_movements.stage_i_causal_admission import Causal21dAdmissionSpec


def _frame(rows: int = 240, *, start: str = "2024-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(7)
    decision = pd.date_range(start, periods=rows, freq="h", tz="UTC")
    side = np.where(np.arange(rows) % 2, "long", "short")
    raw = np.clip(rng.normal(0.0, 0.35, rows), -0.95, 0.95)
    p_clear = np.clip(0.45 + raw * 0.25, .05, .9)
    p_adverse = np.clip(0.35 - raw * .2, .05, .85)
    p_weak = 1.0 - p_clear - p_adverse
    p_weak = np.maximum(p_weak, .02)
    total = p_clear + p_adverse + p_weak
    p_clear, p_adverse, p_weak = p_clear / total, p_adverse / total, p_weak / total
    context = rng.normal(size=rows)
    net = raw * 170.0 + context * 30.0 + rng.normal(0, 25, rows)
    return pd.DataFrame({
        "candidate_id": [f"c{i}" for i in range(rows)], "symbol": np.where(np.arange(rows) % 3, "BTC", "ETH"),
        "side_name": side, "signal_close_ts": decision - pd.Timedelta(hours=1), "decision_ts": decision,
        "label_available_ts": decision + pd.Timedelta(hours=12), "exact_net_bps": net,
        "exact_gross_bps": net + 100.0, "base_direct_score": raw,
        "base_state_p0": p_adverse, "base_state_p1": p_weak, "base_state_p2": p_clear,
        "base_output_entropy": .8, "base_output_top2_margin": .2, "base_output_max_probability": .6,
        "base_strict_oof_available": True, "context": context, "path_descriptor": context + rng.normal(0, .1, rows),
    })


def _spec(*, score_domain: tuple[float, float] = (-1.0, 1.0)) -> StageIIDirectFQ3Spec:
    return StageIIDirectFQ3Spec(
        meta_feature_cols=("context",),
        model_params={"n_estimators": 12, "learning_rate": .1, "num_leaves": 7, "verbosity": -1, "random_state": 3},
        min_train_rows=24, n_validation_folds=3,
        score_domain=score_domain,
        admission_spec=Causal21dAdmissionSpec(min_reference_rows=20, min_side_reference_rows=8, bins=4),
    )


def _candidate() -> StageIIDirectFQ3Candidate:
    return StageIIDirectFQ3Candidate(
        "path3",
        StageIIMetaArchetypeConfig(path_descriptor_cols=("path_descriptor",), components=3, min_side_rows=12, min_component_rows=4, min_train_rows=24, oof_folds=3),
        ("context",),
    )


def _native_state_frame(width: int, rows: int = 240) -> pd.DataFrame:
    """Emulate the 2-state scalar and 5-state ordinal Stage-I handoffs."""
    assert width in (2, 5)
    result = _frame(rows).drop(columns=["base_state_p0", "base_state_p1", "base_state_p2"])
    rng = np.random.default_rng(100 + width)
    raw = rng.uniform(0.0, 1.0, size=(rows, width))
    states = raw / raw.sum(axis=1, keepdims=True)
    for index in range(width):
        result[f"base_state_p{index}"] = states[:, index]
    # Scalar and ordinal Stage-I contracts both emit a native [0, 1] score.
    result["base_direct_score"] = np.clip(
        0.5 * (pd.to_numeric(result.base_direct_score, errors="coerce").to_numpy(float) + 1.0),
        0.0,
        1.0,
    )
    return result


def test_direct_bridge_keeps_native_input_and_maps_only_after_fq3_reconstruction() -> None:
    result = run_stage_ii_direct_fq3_archetype_funnel(_frame(), spec=_spec(), candidates=(_candidate(),))
    # The synthetic data does not prove an archetype gain.  Crucially, the
    # bridge must retain Stage I rather than selecting the ``none`` control.
    assert result.selected_arm is None
    assert result.manifest["decision"] == "NO_STAGE_II_ARCHETYPE_ADVANCES"
    assert result.candidate_audit.loc[0, "matching_control_rows"] == result.candidate_audit.loc[0, "matching_soft_rows"]
    output = next(arm.oof_predictions for arm in result.arms if arm.arm == "soft_memberships")
    scored = output.loc[output.meta_strict_oof_available]
    assert len(scored) and scored.meta_direct_score.notna().all()
    assert np.allclose(scored[["meta_p_error_tercile_0", "meta_p_error_tercile_1", "meta_p_error_tercile_2"]].sum(axis=1), 1.0)
    assert "prequential_base_expected_net_bps" not in result.selected_features
    top = next(arm.metrics for arm in result.arms if arm.arm == "soft_memberships")
    assert top.ranking_basis.eq("pooled_global_after_causal_common_bps_mapping_never_per_timestamp").all()


def test_direct_bridge_rejects_premapped_bps_in_meta_contract() -> None:
    with pytest.raises(StageIIDirectFQ3Error, match="mapped/common-bps"):
        StageIIDirectFQ3Spec(
            meta_feature_cols=("prequential_base_expected_net_bps",), model_params={}, min_train_rows=24
        ).validate()


def test_frozen_direct_scorer_uses_only_prior_resolved_oof_reference_for_mapping() -> None:
    full = _frame(1_100)
    # The same raw identifier is intentionally used once per side.  Mapping
    # must use side::candidate_id, never candidate_id alone.
    full["candidate_id"] = [f"shared-{index // 2}" for index in range(len(full))]
    train = full.iloc[:800].copy()
    # A label-resolution gap makes all frozen training labels available before
    # the later evaluation starts.
    evaluation = full.iloc[1_000:].copy()
    reference = train.loc[:, ["candidate_id", "side_name", "decision_ts", "label_available_ts", "exact_net_bps"]].copy()
    reference["meta_direct_score"] = train.base_direct_score.to_numpy()
    reference["meta_strict_oof_available"] = True
    output = score_frozen_stage_ii_direct_fq3(
        training=train, mapping_reference=reference, evaluation=evaluation,
        spec=_spec(), candidate=_candidate(), selected_arm="soft_memberships",
    )
    assert output.prequential_joint_expected_net_bps.notna().all()
    assert output.meta_causal_21d_expected_net_bps.equals(output.prequential_joint_expected_net_bps)
    assert output.joint_expected_net_bps_semantics.eq(
        "direct_fq3_reconstructed_causal_21d_common_bps_v1"
    ).all()
    # Retained only for old readers; new direct-FQ3 consumers must use joint.
    assert output.prequential_base_expected_net_bps.equals(output.prequential_joint_expected_net_bps)
    assert output.prequential_base_expected_net_bps_semantics.eq(
        "deprecated_compatibility_alias_of_prequential_joint_expected_net_bps"
    ).all()
    assert output.causal_21d_admission_window_days.eq(21).all()
    assert output.meta_score_semantics.eq("same_side_direct_base_output_correctness_q33_v1").all()
    assert np.allclose(output[["meta_p_error_tercile_0", "meta_p_error_tercile_1", "meta_p_error_tercile_2"]].sum(axis=1), 1.0)


@pytest.mark.parametrize("width", (2, 5))
def test_direct_bridge_accepts_native_scalar_and_ordinal_state_contracts(width: int) -> None:
    frame = _native_state_frame(width)
    spec = _spec(score_domain=(0.0, 1.0))
    validated = validate_stage_ii_direct_fq3_ledger(frame, spec=spec)
    state_columns = tuple(f"base_state_p{index}" for index in range(width))
    assert all(name in validated for name in state_columns)

    result = run_stage_ii_direct_fq3_archetype_funnel(
        validated, spec=spec, candidates=(_candidate(),)
    )
    control = next(arm for arm in result.arms if arm.arm == "none")
    assert state_columns == tuple(name for name in control.feature_names if name.startswith("base_state_p"))
    assert "prequential_base_expected_net_bps" not in control.feature_names
    assert result.manifest["base_state_columns"] == list(state_columns)


@pytest.mark.parametrize("width", (2, 5))
def test_frozen_direct_scorer_does_not_relabel_non_r3_states(width: int) -> None:
    full = _native_state_frame(width, rows=1_100)
    full["candidate_id"] = [f"shared-{index // 2}" for index in range(len(full))]
    train = full.iloc[:800].copy()
    evaluation = full.iloc[1_000:].copy()
    reference = train.loc[:, [
        "candidate_id", "side_name", "decision_ts", "label_available_ts", "exact_net_bps",
    ]].copy()
    reference["meta_direct_score"] = train.base_direct_score.to_numpy()
    reference["meta_strict_oof_available"] = True

    output = score_frozen_stage_ii_direct_fq3(
        training=train,
        mapping_reference=reference,
        evaluation=evaluation,
        spec=_spec(score_domain=(0.0, 1.0)),
        candidate=_candidate(),
        selected_arm="none",
    )
    state_columns = tuple(f"base_state_p{index}" for index in range(width))
    assert all(name in output for name in state_columns)
    assert not {"r3_p_adverse", "r3_p_weak", "r3_p_clear"}.intersection(output.columns)
    assert output.prequential_joint_expected_net_bps.notna().all()


def test_direct_bridge_rejects_sparse_or_unsupported_native_state_contract() -> None:
    sparse = _native_state_frame(2).rename(columns={"base_state_p1": "base_state_p2"})
    with pytest.raises(StageIIDirectFQ3Error, match="contiguous 2-, 3-, or 5-state"):
        validate_stage_ii_direct_fq3_ledger(sparse, spec=_spec(score_domain=(0.0, 1.0)))

    unsupported = _native_state_frame(5).drop(columns="base_state_p4")
    with pytest.raises(StageIIDirectFQ3Error, match="contiguous 2-, 3-, or 5-state"):
        validate_stage_ii_direct_fq3_ledger(unsupported, spec=_spec(score_domain=(0.0, 1.0)))


@pytest.mark.parametrize("width", (2, 3, 5))
def test_handoff_materializer_is_identity_bound_and_preserves_dynamic_state_contract(
    tmp_path, width: int,
) -> None:
    ledger = _frame() if width == 3 else _native_state_frame(width)
    if width == 3:
        # This is the frozen R3 writer's historical state spelling.  The
        # materializer must normalise it to the dynamic native-state contract,
        # rather than assume a 3-state caller has already done so.
        ledger["r3_p_adverse"] = ledger["base_state_p0"]
        ledger["r3_p_weak"] = ledger["base_state_p1"]
        ledger["r3_p_clear"] = ledger["base_state_p2"]
        ledger["r3_opportunity_score"] = ledger["base_direct_score"]
        ledger = ledger.drop(columns=["base_state_p0", "base_state_p1", "base_state_p2", "base_direct_score"])
    score_domain = (-1.0, 1.0) if width == 3 else (0.0, 1.0)
    states = tuple(f"base_state_p{index}" for index in range(width))
    # Deliberately give the frozen panel unusable trust values: the materializer
    # must derive those fields from the strict-OOF native simplex instead.
    panel = ledger.loc[:, ["candidate_id", "side_name", "symbol", "signal_close_ts", "context"]].rename(
        columns={"symbol": "__symbol__", "signal_close_ts": "__ts__"}
    )
    panel["base_output_entropy"] = -99.0
    selected_meta = ("context", "base_raw_score", *states, "base_output_entropy", "base_output_top2_margin", "base_output_max_probability")
    root = materialize_stage_ii_direct_fq3_handoff(
        tmp_path / f"handoff_{width}",
        frozen_stage_i_input_panel=panel,
        joint_oof_ledger=ledger,
        selected_causal_feature_cols=("context",),
        selected_meta_feature_cols=selected_meta,
    )
    output = pd.read_parquet(root / "stage_ii_direct_fq3_input.parquet")
    manifest = json.loads((root / "run_manifest.json").read_text())
    checksums = json.loads((root / "checksums.json").read_text())
    validated = validate_stage_ii_direct_fq3_ledger(output, spec=_spec(score_domain=score_domain))
    assert tuple(name for name in validated if name.startswith("base_state_p")) == states
    assert output.__symbol__.astype(str).equals(output.symbol.astype(str))
    assert pd.to_datetime(output.__ts__, utc=True).equals(pd.to_datetime(output.signal_close_ts, utc=True))
    assert not np.allclose(output.base_output_entropy.to_numpy(float), -99.0)
    assert manifest["base_state_width"] == width
    assert manifest["identity_join"] == "exact_one_to_one_candidate_id_side_symbol_signal_close_ts"
    assert manifest["trust_lineage"] == "derived_from_joint_oof_native_base_simplex_not_input_panel"
    assert set(checksums) == {"stage_ii_direct_fq3_input.parquet", "run_manifest.json"}


def test_handoff_materializer_rejects_nonidentical_panel_and_joint_oof_identity(tmp_path) -> None:
    ledger = _frame()
    panel = ledger.loc[:, ["candidate_id", "side_name", "symbol", "signal_close_ts", "context"]].rename(
        columns={"symbol": "__symbol__", "signal_close_ts": "__ts__"}
    )
    panel.loc[0, "candidate_id"] = "not-the-joint-oof-row"
    with pytest.raises(StageIIDirectFQ3Error, match="non-identical exact identities"):
        materialize_stage_ii_direct_fq3_handoff(
            tmp_path / "identity_drift",
            frozen_stage_i_input_panel=panel,
            joint_oof_ledger=ledger,
            selected_causal_feature_cols=("context",),
            selected_meta_feature_cols=("context", "base_raw_score", "base_state_p0", "base_state_p1", "base_state_p2", "base_output_entropy", "base_output_top2_margin", "base_output_max_probability"),
        )
