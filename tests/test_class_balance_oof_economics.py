from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.catboost_archetype_classifier import (
    PathArchetypeConfig,
    rematerialize_final_class_balance_params,
)
from extreme_price_movements.class_balance_oof_economics import (
    BalanceArmOOF,
    EconomicOOFConfig,
    score_class_balance_oof_economics,
)

CLASSES = ("adverse", "favourable")
ARMS = (
    "uniform",
    "frequency_power_0.25",
    "frequency_power_0.50",
    "frequency_power_0.75",
)


def _frame() -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=12, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "candidate_id": [f"candidate-{index}" for index in range(12)],
            "__ts__": dates,
            "__label_end_ts__": dates + pd.Timedelta(hours=6),
            "__side__": ["long"] * 12,
            "path_arch_final_return_net_1pct": [
                -1,
                -1,
                -1,
                -1,
                1,
                1,
                1,
                1,
                -1,
                1,
                -1,
                1,
            ],
            "path_arch_peak_mfe_atr": [0.2, 0.3, 0.2, 0.3, 2, 2, 2, 2, 0.2, 2, 0.2, 2],
            "path_arch_mae_12h_r": [
                -1.0,
                -0.8,
                -1.1,
                -0.9,
                -0.1,
                -0.2,
                -0.1,
                -0.2,
                -1.0,
                -0.2,
                -1.1,
                -0.2,
            ],
            "path_arch_mae_before_meaningful_mfe_r": [
                1.0,
                0.8,
                1.1,
                0.9,
                0.1,
                0.2,
                0.1,
                0.2,
                1.0,
                0.2,
                1.1,
                0.2,
            ],
            "path_arch_stop_before_meaningful_mfe": [
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                0,
                1,
                0,
                1,
                0,
            ],
            "path_arch_reaches_meaningful_mfe": [0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1],
            "path_arch_time_to_first_meaningful_mfe_h": [
                12,
                12,
                12,
                12,
                2,
                2,
                2,
                2,
                12,
                2,
                12,
                2,
            ],
            "path_arch_peak_retention_ratio": [
                -0.5,
                -0.4,
                -0.5,
                -0.4,
                0.8,
                0.9,
                0.8,
                0.9,
                -0.5,
                0.8,
                -0.5,
                0.9,
            ],
            "path_arch_time_to_trailing_h": [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                3,
                3,
                3,
                3,
                np.nan,
                3,
                np.nan,
                3,
            ],
            "path_arch_mfe_to_activation_distance": [0.3] * 12,
        }
    )


def _folds() -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            fold_id=0,
            train_indices=np.array([0, 1, 2]),
            validation_indices=np.array([4, 5, 6, 7]),
        ),
        SimpleNamespace(
            fold_id=1,
            train_indices=np.array([0, 1, 2, 3, 4, 5, 6]),
            validation_indices=np.array([8, 9, 10, 11]),
        ),
    ]


def _arm(
    probabilities: np.ndarray, *, feature: str = "feature-v1", guard: bool = True
) -> BalanceArmOOF:
    return BalanceArmOOF(
        probabilities=probabilities,
        fold_ids=np.array([-1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1]),
        folds=_folds(),
        classes=CLASSES,
        structural_fingerprint="hpo-v1",
        feature_fingerprint=feature,
        geometry_fingerprint="geometry-v1",
        oof_guard={"passed": guard},
        row_ids=np.array([f"candidate-{index}" for index in range(12)]),
    )


def _probabilities(good: bool) -> np.ndarray:
    result = np.full((12, 2), [0.5, 0.5], dtype=float)
    truth = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    result[:, 0] = np.where(truth == 0, 0.85 if good else 0.55, 0.15 if good else 0.45)
    result[:, 1] = 1.0 - result[:, 0]
    return result


def _arms(*, guard=True) -> dict[str, BalanceArmOOF]:
    uniform = _probabilities(True)
    # Candidate has equal ML quality but changes only the OOS ranking within
    # each class; it cannot satisfy the strict all-economic gates accidentally.
    return {
        "uniform": _arm(uniform, guard=guard),
        "frequency_power_0.25": _arm(uniform.copy(), guard=guard),
        "frequency_power_0.50": _arm(uniform.copy(), guard=guard),
        "frequency_power_0.75": _arm(uniform.copy(), guard=guard),
    }


def test_scores_only_with_train_only_fold_priors_and_emits_consumable_uniform_default() -> (
    None
):
    frame = _frame()
    result = score_class_balance_oof_economics(
        frame,
        np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1]),
        _arms(),
        config=EconomicOOFConfig(minimum_month_rows=1),
    )
    assert result["selection_provenance"]["arm"] == "uniform"
    assert result["selection_provenance"]["promotion_eligible"] is True
    assert (
        result["selection_provenance"]["schema"]
        == "catboost_path_archetype_class_balance_arm_selection_v1"
    )
    first_prior = result["per_arm"]["uniform"]["train_only_priors"][0]
    assert first_prior["source"] == "exact_purged_fold_train_indices_only"
    assert first_prior["targets"]["net_ev"]["support"] == [3, 0]
    assert result["per_arm"]["uniform"]["aggregate"]["economic"]["net_ev"]["rows"] == 8
    assert (
        result["per_arm"]["uniform"]["aggregate"]["economic"][
            "mae_before_meaningful_r"
        ]["rows"]
        == 8
    )
    assert result["per_arm"]["uniform"]["months"][0]["supported"] is True
    assert result["contract"]["selector_config"]["identity_col"] == "candidate_id"
    assert len(result["contract"]["selector_config_sha256"]) == 64


def test_rejects_nonidentical_feature_contract_before_scoring() -> None:
    arms = _arms()
    arms["frequency_power_0.50"] = _arm(
        _probabilities(True), feature="different", guard=True
    )
    with pytest.raises(ValueError, match="matched structural/feature/geometry"):
        score_class_balance_oof_economics(
            _frame(),
            np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1]),
            arms,
            config=EconomicOOFConfig(minimum_month_rows=1),
        )


def test_guard_failure_is_explicitly_nonpromotable_uniform() -> None:
    result = score_class_balance_oof_economics(
        _frame(),
        np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1]),
        _arms(guard=False),
        config=EconomicOOFConfig(minimum_month_rows=1),
    )
    provenance = result["selection_provenance"]
    assert provenance["arm"] == "uniform"
    assert provenance["promotion_eligible"] is False
    assert provenance["selection_status"] == "uniform_default_oof_guard_failed"


def test_nan_unscored_probability_rows_are_allowed_and_never_scored() -> None:
    arms = _arms()
    for arm in arms.values():
        arm.probabilities[:4] = np.nan
    result = score_class_balance_oof_economics(
        _frame(),
        np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1]),
        arms,
        config=EconomicOOFConfig(minimum_month_rows=1),
    )
    assert result["per_arm"]["uniform"]["aggregate"]["ml"]["logloss"] < 1.0


def test_covered_uniform_selection_can_be_rematerialized_for_final_fit() -> None:
    labels = np.array(
        [
            "adverse",
            "adverse",
            "adverse",
            "adverse",
            "favourable",
            "favourable",
            "favourable",
            "favourable",
            "adverse",
            "favourable",
            "adverse",
            "favourable",
        ]
    )
    result = score_class_balance_oof_economics(
        _frame(),
        np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1]),
        _arms(),
        config=EconomicOOFConfig(minimum_month_rows=1),
    )
    materialized = rematerialize_final_class_balance_params(
        {
            "class_balance_arm": "uniform",
            "class_balance_selection_provenance": result["selection_provenance"],
        },
        labels,
        config=PathArchetypeConfig(class_order=CLASSES),
    )
    assert materialized["class_balance_provenance"]["arm"] == "uniform"


def test_rejects_unresolved_train_label_even_if_oof_arrays_match() -> None:
    frame = _frame()
    frame.loc[0, "__label_end_ts__"] = frame.loc[4, "__ts__"]
    with pytest.raises(ValueError, match="unresolved labels"):
        score_class_balance_oof_economics(
            frame,
            np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1]),
            _arms(),
            config=EconomicOOFConfig(minimum_month_rows=1),
        )


def test_rejects_reordered_outcome_identity_even_when_other_rows_match() -> None:
    frame = _frame()
    frame["candidate_id"] = frame["candidate_id"].iloc[::-1].to_numpy()
    with pytest.raises(ValueError, match="identities must exactly match"):
        score_class_balance_oof_economics(
            frame,
            np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1]),
            _arms(),
            config=EconomicOOFConfig(minimum_month_rows=1),
        )


def test_rejects_multi_side_input_to_side_local_scorer() -> None:
    frame = _frame()
    frame.loc[11, "__side__"] = "short"
    with pytest.raises(ValueError, match="exactly one canonical side"):
        score_class_balance_oof_economics(
            frame,
            np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1]),
            _arms(),
            config=EconomicOOFConfig(minimum_month_rows=1),
        )
