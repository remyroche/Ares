from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.regime_multiview_selection import (
    MultiviewSelectionConfig,
    infer_multiview_lineage,
    select_fold_local_multiview_features,
)


def _train_frame(rows: int = 180) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    index = pd.Index([f"train_{i:03d}" for i in range(rows)], name="row_id")
    x = np.linspace(-2.0, 2.0, rows)
    frame = pd.DataFrame(
        {
            "mv__alpha__delta_1h": x,
            "mv__alpha__delta_3h": x + 1e-5 * np.sin(x),  # redundancy cluster
            "mv__beta__robust_z_24h": np.sin(2.0 * x),
            "mv__beta__realized_vol_1h": np.abs(x),
            "mv__liquidity__volume_proxy__stress_1h": -x,
            "mv__dependence__eig1_share_24h": np.cos(1.5 * x),
            "mv__dependence__corr_frobenius_shift_72h": np.sin(3.0 * x),
            "mv__alpha__iqr_1h": 1.0,  # fails robust variance gate
            "mv__sparse__delta_1h": np.where(np.arange(rows) < 40, x[:40].mean(), np.nan),
        },
        index=index,
    )
    regime = pd.Series((np.sin(2.0 * x) > 0).astype(float), index=index)
    transition = pd.Series((np.cos(1.5 * x) > 0).astype(float), index=index)
    return frame, regime, transition


def _config() -> MultiviewSelectionConfig:
    return MultiviewSelectionConfig(
        fold_id="train_fold_1",
        min_coverage=0.8,
        max_candidates_per_family_before_redundancy=20,
        family_caps={
            "distribution_dynamics": 6,
            "volatility": 3,
            "liquidity_proxy": 3,
            "dependence_covariance": 4,
            "other": 1,
        },
        max_correlation_rows=100,
        random_state=9,
    )


def test_lineage_infers_family_horizon_and_transform() -> None:
    lineage = infer_multiview_lineage(
        [
            "mv__alpha__delta_3h",
            "mv__alpha__realized_vol_24h",
            "mv__liquidity__volume_proxy__stress_1h",
            "mv__dependence__corr_frobenius_shift_72h",
        ]
    ).set_index("feature")

    assert lineage.loc["mv__alpha__delta_3h", ["family", "horizon"]].tolist() == ["distribution_dynamics", "3h"]
    assert lineage.loc["mv__alpha__realized_vol_24h", "family"] == "volatility"
    assert lineage.loc["mv__liquidity__volume_proxy__stress_1h", "source_field"] == "volume_proxy"
    assert lineage.loc["mv__dependence__corr_frobenius_shift_72h", "family"] == "dependence_covariance"


def test_fold_local_screening_prunes_bad_fields_but_preserves_horizons() -> None:
    frame, regime, transition = _train_frame()
    result = select_fold_local_multiview_features(
        frame,
        config=_config(),
        regime_train_labels=regime,
        transition_train_labels=transition,
        fold_training_row_ids=frame.index,
    )
    lineage = result.lineage.set_index("feature")

    assert "mv__alpha__iqr_1h" not in result.unsupervised_features
    assert "mv__sparse__delta_1h" not in result.unsupervised_features
    assert {"1h", "3h", "24h"}.issubset(set(result.diagnostics["horizons_unsupervised"]))
    assert lineage.loc["mv__alpha__delta_1h", "redundancy_cluster"] == lineage.loc["mv__alpha__delta_3h", "redundancy_cluster"]
    assert lineage.loc["mv__alpha__delta_1h", "unsupervised_selected"]
    assert lineage.loc["mv__alpha__delta_3h", "unsupervised_selected"]
    assert result.regime_features != result.transition_features
    assert result.diagnostics["labels_used_for_unsupervised_selection"] is False


def test_held_out_values_and_labels_cannot_affect_train_only_selection() -> None:
    frame, regime, transition = _train_frame(220)
    train = frame.iloc[:170].copy()
    first = select_fold_local_multiview_features(
        train,
        config=_config(),
        regime_train_labels=regime.iloc[:170],
        transition_train_labels=transition.iloc[:170],
        fold_training_row_ids=train.index,
    )

    held_out_mutated = frame.copy()
    held_out_mutated.iloc[170:] *= -9_999.0
    second = select_fold_local_multiview_features(
        held_out_mutated.iloc[:170],
        config=_config(),
        regime_train_labels=regime.iloc[:170],
        transition_train_labels=transition.iloc[:170],
        fold_training_row_ids=train.index,
    )

    assert first.unsupervised_features == second.unsupervised_features
    assert first.regime_features == second.regime_features
    assert first.transition_features == second.transition_features
    pd.testing.assert_frame_equal(first.lineage, second.lineage)


def test_labels_are_optional_and_train_identity_and_denylist_are_enforced() -> None:
    frame, _, _ = _train_frame()
    result = select_fold_local_multiview_features(
        frame,
        config=_config(),
        fold_training_row_ids=frame.index,
    )
    assert result.regime_features == []
    assert result.transition_features == []

    with pytest.raises(ValueError, match="fold_training_row_ids"):
        select_fold_local_multiview_features(frame, config=_config(), fold_training_row_ids=frame.index[:-1])

    forbidden = frame.copy()
    forbidden["target__future_ev"] = 1.0
    with pytest.raises(ValueError, match="forbidden"):
        select_fold_local_multiview_features(forbidden, config=_config(), fold_training_row_ids=forbidden.index)
