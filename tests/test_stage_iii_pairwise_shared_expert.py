from __future__ import annotations

from hashlib import sha256
import json

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.stage_iii_pairwise_shared_expert import (
    COMMON_BPS_RECONSTRUCTION,
    ONE_SHARED_BOTH_SIDE_MODEL,
    PairwiseSharedExpertError,
    PairwiseSharedResidualConfig,
    fit_pairwise_shared_residual_expert,
    fit_target_preserving_pairwise_adapter,
)
from extreme_price_movements.stage_iii_robust_target_models import (
    RobustTargetModelConfig,
    fit_ordinal_shared_robust_target,
    fit_quantile_shared_robust_target,
)
from extreme_price_movements.stage_iii_residual_target_challengers import (
    PairConstructionConfig,
)


def _frame() -> pd.DataFrame:
    """Four bounded side/date contexts with strict complete labels."""
    records: list[dict[str, object]] = []
    start = pd.Timestamp("2024-01-01 09:00", tz="UTC")
    row = 0
    for day in range(4):
        for side_index, side in enumerate(("long", "short")):
            for within in range(8):
                decision = start + pd.Timedelta(days=day, minutes=within * 5 + side_index)
                # Consecutive rows have 55 bps gaps, so F1 has strictly more
                # eligible information than an F2 100-bps pair selection.
                residual = -192.5 + 55.0 * within + 8.0 * side_index
                base = 32.0 + 2.0 * within - 3.0 * side_index
                prior = -9.0 + 3.0 * day + 2.0 * side_index
                records.append(
                    {
                        "candidate_id": f"c{row:03d}",
                        "symbol": "BTC" if row % 3 else "ETH",
                        "decision_ts": decision,
                        "label_available_ts": decision + pd.Timedelta(hours=12),
                        "side_name": side,
                        "exact_net_bps": base + prior + residual,
                        "prequential_base_expected_net_bps": base,
                        "prequential_soft_regime_prior_residual_bps": prior,
                        "candidate_residual_bps": residual,
                        "cost_to_atr": 0.90 + 0.02 * (within % 3),
                        "p_regime_calm": 0.82 - 0.015 * within,
                        "p_regime_stress": 0.18 + 0.015 * within,
                        # Three deliberately causal, non-outcome feature fields.
                        "feature_signal": float(within) + 0.2 * side_index,
                        "feature_context": float(day) - 0.1 * within,
                        "feature_ood": 0.05 * (within % 4) + 0.03 * day,
                        "base_map_is_prequential": True,
                        "base_map_source_side": side,
                        "base_map_max_label_available_ts": decision - pd.Timedelta(hours=1),
                        "soft_regime_is_causal_prequential": True,
                        "soft_regime_fit_end_ts": decision - pd.Timedelta(minutes=30),
                        "prior_resolved_max_label_available_ts": decision - pd.Timedelta(minutes=15),
                        "cost_atr_is_causal": True,
                    }
                )
                row += 1
    return pd.DataFrame(records)


FEATURES = (
    "feature_signal",
    "feature_context",
    "feature_ood",
    "p_regime_calm",
    "p_regime_stress",
)
SOFT_REGIMES = ("p_regime_calm", "p_regime_stress")
FIT_CUTOFF = "2024-01-08T00:00:00Z"
POINTWISE_PARAMS = {
    "n_estimators": 24,
    "learning_rate": 0.06,
    "num_leaves": 7,
    "min_child_samples": 2,
}
CONFIG = PairwiseSharedResidualConfig(
    pairwise_blend_weight=0.10,
    ranker_estimators=18,
    ranker_learning_rate=0.06,
    ranker_num_leaves=7,
    ranker_min_child_samples=2,
    ranker_l2=2.0,
    random_state=13,
)
PAIR_CONFIG = PairConstructionConfig(
    max_base_ev_difference_bps=30.0,
    max_cost_atr_difference=0.10,
    max_pairs_per_better_row=3,
    max_rows_per_side_date=64,
)


def _fit(arm: str, frame: pd.DataFrame | None = None):
    return fit_pairwise_shared_residual_expert(
        _frame() if frame is None else frame,
        arm=arm,
        feature_names=FEATURES,
        soft_regime_columns=SOFT_REGIMES,
        fit_before_utc=FIT_CUTOFF,
        pair_config=PAIR_CONFIG,
        config=CONFIG,
        pointwise_params=POINTWISE_PARAMS,
    )


def _digest(value: object) -> str:
    return sha256(json.dumps(value, default=str, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class _FakeFrozenT4Base:
    """A non-refittable target base used to test the generic adapter boundary."""

    def __init__(self, frame: pd.DataFrame, *, feature_names: tuple[str, ...] = FEATURES) -> None:
        self.feature_names = feature_names
        target = frame["candidate_residual_bps"].to_numpy(float)
        self.audit = {
            "arm": "T4_quantile",
            "formulation": "fake_frozen_quantile_target",
            "routing": ONE_SHARED_BOTH_SIDE_MODEL,
            "reconstruction": COMMON_BPS_RECONSTRUCTION,
            "feature_names": list(feature_names),
            "feature_sha256": _digest(list(feature_names)),
            "training_row_count": len(frame),
            "training_candidate_ids_sha256": _digest(frame["candidate_id"].astype(str).tolist()),
            "training_cutoff_utc": FIT_CUTOFF,
            "max_label_available_utc": pd.Timestamp(frame["label_available_ts"].max()).isoformat(),
            "target_label_sha256": _digest(np.round(target, 8).tolist()),
        }

    def predict_candidate_residual_bps(self, frame: pd.DataFrame) -> np.ndarray:
        # Deliberately avoid labels: this is a frozen, causal fake target score.
        return (
            12.0 * frame["feature_signal"].to_numpy(float)
            - 4.0 * frame["feature_context"].to_numpy(float)
        ).astype(np.float64)


def test_f0_is_one_shared_both_side_pointwise_bps_expert_with_exact_audit() -> None:
    frame = _frame()
    fit = _fit("F0_pointwise", frame)

    audit = fit.audit
    assert audit.arm == "F0_pointwise"
    assert audit.routing == ONE_SHARED_BOTH_SIDE_MODEL
    assert audit.reconstruction == COMMON_BPS_RECONSTRUCTION
    assert audit.training_row_count == len(frame)
    assert dict(audit.training_rows_by_side) == {"long": 32, "short": 32}
    assert audit.feature_names == FEATURES
    assert len(audit.feature_sha256) == 64
    assert audit.training_cutoff_utc == pd.Timestamp(FIT_CUTOFF)
    assert audit.max_label_available_utc < pd.Timestamp(FIT_CUTOFF)
    assert audit.pairwise_model_class is None
    assert audit.pairwise_calibration is None
    assert audit.pair_support.pair_selection == "disabled_for_F0_pointwise"
    assert audit.pair_support.selected_pairs == 0

    residual = fit.predict_candidate_residual_bps(frame)
    expected = fit.predict_expected_net_bps(frame)
    np.testing.assert_allclose(
        expected,
        frame["prequential_base_expected_net_bps"].to_numpy()
        + frame["prequential_soft_regime_prior_residual_bps"].to_numpy()
        + residual,
    )
    payload = audit.to_dict()
    assert payload["pair_support"]["selected_pairs"] == 0
    assert payload["training_cutoff_utc"].endswith("+00:00")


@pytest.mark.parametrize(
    "arm, separation",
    [("F1_pairwise_50bps", 50.0), ("F2_pairwise_100bps", 100.0)],
)
def test_pairwise_arms_use_only_bounded_context_pairs_and_reconstruct_common_bps(
    arm: str, separation: float
) -> None:
    frame = _frame()
    fit = _fit(arm, frame)

    audit = fit.audit
    support = audit.pair_support
    assert audit.routing == ONE_SHARED_BOTH_SIDE_MODEL
    assert audit.pairwise_model_class == "LGBMRanker"
    assert support.separation_bps == separation
    assert support.constructed_pairs >= support.selected_pairs > 0
    assert support.selected_pair_rows == 2 * support.selected_pairs
    assert support.selected_unique_candidates > 0
    assert support.max_pair_label_available_utc is not None
    assert support.max_pair_label_available_utc < audit.training_cutoff_utc
    assert support.pair_builder_routing == "one_shared_model_no_local_experts"
    assert len(str(support.selected_pair_ledger_sha256)) == 64
    assert set(dict(support.selected_pairs_by_side)) == {"long", "short"}
    assert audit.pairwise_calibration is not None
    assert audit.pairwise_calibration.rows == len(frame)
    assert audit.pairwise_calibration.fit_before_utc == pd.Timestamp(FIT_CUTOFF)

    residual = fit.predict_candidate_residual_bps(frame)
    expected = fit.predict_expected_net_bps(frame)
    assert np.isfinite(residual).all()
    np.testing.assert_allclose(
        expected,
        frame["prequential_base_expected_net_bps"].to_numpy()
        + frame["prequential_soft_regime_prior_residual_bps"].to_numpy()
        + residual,
    )


def test_pairwise_fit_is_deterministic_for_identical_frozen_support() -> None:
    frame = _frame()
    left = _fit("F1_pairwise_50bps", frame)
    right = _fit("F1_pairwise_50bps", frame)

    assert left.audit.feature_sha256 == right.audit.feature_sha256
    assert (
        left.audit.pair_support.selected_pair_ledger_sha256
        == right.audit.pair_support.selected_pair_ledger_sha256
    )
    np.testing.assert_allclose(
        left.predict_candidate_residual_bps(frame),
        right.predict_candidate_residual_bps(frame),
        rtol=0.0,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "mutation, message",
    [
        (
            lambda frame: frame.assign(
                label_available_ts=pd.Timestamp(FIT_CUTOFF)
            ),
            "unresolved/current/future",
        ),
        (lambda frame: frame.assign(base_map_is_prequential=False), "base_map_is_prequential"),
        (
            lambda frame: frame.assign(candidate_residual_bps=0.0),
            "must equal exact_net_bps",
        ),
        (lambda frame: frame.assign(side_name="long"), "both-side"),
    ],
)
def test_round_f_rejects_unresolved_noncausal_or_nonshared_training_support(
    mutation, message: str
) -> None:
    with pytest.raises(PairwiseSharedExpertError, match=message):
        _fit("F1_pairwise_50bps", mutation(_frame()))


@pytest.mark.parametrize("bad_value", ["False", np.nan, 2, -1])
def test_round_f_rejects_truthy_non_boolean_lineage(bad_value: object) -> None:
    frame = _frame()
    frame["base_map_is_prequential"] = bad_value
    with pytest.raises(PairwiseSharedExpertError, match="explicit true booleans"):
        _fit("F0_pointwise", frame)


def test_round_f_rejects_hard_regime_or_missing_frozen_inference_feature() -> None:
    frame = _frame()
    with pytest.raises(PairwiseSharedExpertError, match="hard regime"):
        fit_pairwise_shared_residual_expert(
            frame.assign(hard_regime_id=0.0),
            arm="F0_pointwise",
            feature_names=(*FEATURES, "hard_regime_id"),
            soft_regime_columns=SOFT_REGIMES,
            fit_before_utc=FIT_CUTOFF,
            pointwise_params=POINTWISE_PARAMS,
        )

    fit = _fit("F1_pairwise_50bps", frame)
    with pytest.raises(PairwiseSharedExpertError, match="missing frozen features"):
        fit.predict_candidate_residual_bps(frame.drop(columns=["feature_ood"]))


def test_pairwise_config_must_predeclare_its_selected_threshold() -> None:
    with pytest.raises(PairwiseSharedExpertError, match="predeclare"):
        fit_pairwise_shared_residual_expert(
            _frame(),
            arm="F2_pairwise_100bps",
            feature_names=FEATURES,
            soft_regime_columns=SOFT_REGIMES,
            fit_before_utc=FIT_CUTOFF,
            pair_config=PairConstructionConfig(separation_bps=(50.0,)),
            config=CONFIG,
            pointwise_params=POINTWISE_PARAMS,
        )


def test_target_preserving_f0_wraps_fake_t4_without_changing_its_prediction() -> None:
    frame = _frame()
    base = _FakeFrozenT4Base(frame)
    adapter = fit_target_preserving_pairwise_adapter(
        frame,
        base_model=base,
        arm="F0_pointwise",
        feature_names=FEATURES,
        soft_regime_columns=SOFT_REGIMES,
        fit_before_utc=FIT_CUTOFF,
        pair_config=PAIR_CONFIG,
        config=CONFIG,
    )
    expected_base = base.predict_candidate_residual_bps(frame)
    actual = adapter.predict_candidate_residual_bps(frame)
    # F0 must be a no-op, including retaining the base output dtype.
    assert actual.dtype == expected_base.dtype
    np.testing.assert_array_equal(actual, expected_base)
    audit = adapter.audit
    assert audit.pairwise_model_class is None
    assert audit.pair_support.selected_pairs == 0
    assert audit.preserved_base_target.base_target_arm == "T4_quantile"
    assert audit.preserved_base_target.base_training_prediction_sha256 == _digest(
        np.round(expected_base, 8).tolist()
    )


@pytest.mark.parametrize("base_kind, arm", [("ordinal", "F1_pairwise_50bps"), ("quantile", "F2_pairwise_100bps")])
def test_target_preserving_adapter_wraps_actual_t3_and_t4_without_refitting_target(
    base_kind: str, arm: str
) -> None:
    frame = _frame()
    target_config = RobustTargetModelConfig(
        n_estimators=18,
        learning_rate=0.06,
        num_leaves=7,
        min_child_samples=2,
        l2_regularization=2.0,
        random_state=41,
    )
    base = (
        fit_ordinal_shared_robust_target(
            frame, feature_names=FEATURES, fit_before_utc=FIT_CUTOFF, config=target_config
        )
        if base_kind == "ordinal"
        else fit_quantile_shared_robust_target(
            frame, feature_names=FEATURES, fit_before_utc=FIT_CUTOFF, config=target_config
        )
    )
    adapter = fit_target_preserving_pairwise_adapter(
        frame,
        base_model=base,
        arm=arm,
        feature_names=FEATURES,
        soft_regime_columns=SOFT_REGIMES,
        fit_before_utc=FIT_CUTOFF,
        pair_config=PAIR_CONFIG,
        config=CONFIG,
    )
    assert adapter.audit.preserved_base_target.base_target_arm == base.audit.arm
    assert adapter.audit.preserved_base_target.base_audit_sha256 == _digest(base.audit.to_dict())
    assert adapter.audit.pairwise_model_class == "LGBMRanker"
    assert adapter.audit.pairwise_calibration is not None
    adjusted = adapter.predict_candidate_residual_bps(frame)
    assert np.isfinite(adjusted).all()
    expected = adapter.predict_expected_net_bps(frame)
    np.testing.assert_allclose(
        expected,
        frame["prequential_base_expected_net_bps"].to_numpy()
        + frame["prequential_soft_regime_prior_residual_bps"].to_numpy()
        + adjusted,
    )


def test_target_preserving_adapter_rejects_base_target_digest_or_contract_mismatch() -> None:
    frame = _frame()
    base = _FakeFrozenT4Base(frame)
    base.audit["target_label_sha256"] = "not-the-candidate-target"
    with pytest.raises(PairwiseSharedExpertError, match="target-label digest"):
        fit_target_preserving_pairwise_adapter(
            frame,
            base_model=base,
            arm="F0_pointwise",
            feature_names=FEATURES,
            soft_regime_columns=SOFT_REGIMES,
            fit_before_utc=FIT_CUTOFF,
            pair_config=PAIR_CONFIG,
            config=CONFIG,
        )

    substituted = _FakeFrozenT4Base(frame)
    substituted.audit["arm"] = "T0_huber"
    with pytest.raises(PairwiseSharedExpertError, match="target substitution"):
        fit_target_preserving_pairwise_adapter(
            frame,
            base_model=substituted,
            arm="F0_pointwise",
            feature_names=FEATURES,
            soft_regime_columns=SOFT_REGIMES,
            fit_before_utc=FIT_CUTOFF,
            pair_config=PAIR_CONFIG,
            config=CONFIG,
        )
