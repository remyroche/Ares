from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.run_meaningful_mfe_exact_grid_cost_buffer_sensitivity import (
    BUFFERS_BPS,
    compose_opportunity_capture,
    derive_cost_buffer_targets,
    economic_metrics,
    load_frozen_geometry,
)
from scripts.run_meaningful_mfe_exact_grid_reset import (
    JULY_DIAGNOSTIC_END,
    JULY_START,
    july_grouped_day_folds,
)


def _economics_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c", "d"],
            "__ts__": pd.date_range("2026-07-01", periods=4, freq="h", tz="UTC"),
            "__symbol__": ["A", "B", "A", "B"],
            "side_name": ["long", "short", "long", "short"],
            "execution_mfe_return_12h": [0.0100, 0.0200, 0.0300, 0.0050],
            "execution_cost_return": [0.0100, 0.0100, 0.0100, 0.0100],
            "execution_gross_ev_12h": [0.0100, 0.0250, 0.0400, -0.0100],
            "execution_net_ev_12h": [0.0000, 0.0150, 0.0300, -0.0200],
        }
    )


def test_row_cost_buffer_labels_are_strict_and_exact_policy_based() -> None:
    result = derive_cost_buffer_targets(_economics_frame(), 0)

    # The first row ties both boundaries and must not be upgraded into a win.
    assert result["opportunity_0bps"].tolist() == [0, 1, 1, 0]
    assert result["capture_0bps"].tolist() == [0, 1, 1, 0]
    at_25 = derive_cost_buffer_targets(_economics_frame(), 25)
    assert at_25["opportunity_25bps"].tolist() == [0, 1, 1, 0]
    assert at_25["capture_25bps"].tolist() == [0, 1, 1, 0]
    assert set(BUFFERS_BPS) == {0, 25, 50, 100}


def test_cost_identity_finiteness_and_capture_implication_are_enforced() -> None:
    bad_cost = _economics_frame()
    bad_cost.loc[0, "execution_gross_ev_12h"] = 0.02
    with pytest.raises(ValueError, match="gross-cost-net"):
        derive_cost_buffer_targets(bad_cost, 0)

    bad_mfe = _economics_frame()
    bad_mfe.loc[0, "execution_mfe_return_12h"] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        derive_cost_buffer_targets(bad_mfe, 0)

    impossible = _economics_frame()
    impossible.loc[3, ["execution_mfe_return_12h", "execution_gross_ev_12h", "execution_net_ev_12h"]] = [0.001, 0.04, 0.03]
    with pytest.raises(ValueError, match="must imply"):
        derive_cost_buffer_targets(impossible, 0)


def test_conditional_composition_is_product_and_rejects_invalid_probabilities() -> None:
    assert np.allclose(compose_opportunity_capture([0.8, 0.2], [0.5, 0.7]), [0.4, 0.14])
    with pytest.raises(ValueError, match="bounded"):
        compose_opportunity_capture([1.1], [0.5])
    with pytest.raises(ValueError, match="shape"):
        compose_opportunity_capture([0.5], [0.5, 0.5])


def test_grouped_july_folds_remain_contiguous_partition_with_source_embargo() -> None:
    panel = pd.DataFrame({"__ts__": pd.date_range(JULY_START, JULY_DIAGNOSTIC_END, freq="6h", inclusive="left")})
    folds = july_grouped_day_folds(panel)
    covered: list[int] = []
    for _, train, validation, days in folds:
        covered.extend(validation.tolist())
        train_time = panel.iloc[train]["__ts__"]
        for day in days:
            start = pd.Timestamp(day, tz="UTC") - pd.Timedelta(hours=12)
            end = pd.Timestamp(day, tz="UTC") + pd.Timedelta(days=1, hours=12)
            assert not (train_time.ge(start) & train_time.lt(end)).any()
    assert sorted(covered) == list(range(len(panel)))


def test_global_top_k_is_deterministic_and_reports_side_composition() -> None:
    frame = _economics_frame()
    frame["score"] = 1.0
    frame = derive_cost_buffer_targets(frame, 0)
    first = economic_metrics(
        frame.sample(frac=1.0, random_state=1),
        "score",
        buffer_bps=0,
        fraction=0.5,
        scope="test",
        side="pooled",
    )
    second = economic_metrics(
        frame.sample(frac=1.0, random_state=2),
        "score",
        buffer_bps=0,
        fraction=0.5,
        scope="test",
        side="pooled",
    )
    assert first["selected_rows"] == second["selected_rows"] == 2
    assert first["net_ev_bps"] == second["net_ev_bps"]
    assert first["long_selected_rows"] == 1
    assert first["short_selected_rows"] == 1
    assert first["opportunity_precision"] == 0.5
    assert first["capture_precision"] == 0.5
    assert first["loss_rate"] == 0.0


def test_primary_v2_geometry_provenance_is_hash_bound_and_geometry_only() -> None:
    geometry, provenance = load_frozen_geometry()
    assert provenance["geometry_only"] is True
    assert provenance["features_reselected_train_only"] is True
    assert len(provenance["report_sha256"]) == 64
    for family in geometry.values():
        for side in family.values():
            for head in side.values():
                assert head["feature_count"] > 0
                assert head["params"]
                assert "selected_features" not in head
