from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.path_archetype_economic_ic import (
    EconomicICConfig,
    PathArchetypeEconomicIC,
)


CLASSES = ("good", "neutral", "bad")


def _train_frame() -> pd.DataFrame:
    rows = []
    values = {
        "good": (2.0, 3.0, 0.2, 1.0, 0.0),
        "neutral": (0.0, 1.5, 1.0, 4.0, 0.0),
        "bad": (-2.0, 0.2, 3.0, 9.0, 1.0),
    }
    for class_name, outcome in values.items():
        for offset in (-0.1, 0.0, 0.1):
            rows.append(
                {
                    "path_geometry_label": class_name,
                    "net_ev_after_1pct_return": outcome[0] + offset,
                    "peak_mfe_12h_atr": outcome[1] + offset,
                    "mae_12h_atr": max(outcome[2] - offset, 0.01),
                    "time_to_first_meaningful_mfe": outcome[3] - offset,
                    "stop_probability": outcome[4],
                }
            )
    return pd.DataFrame(rows)


def _oos_frame() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    labels = ["good", "neutral", "bad", "good"] * 3
    rows = []
    for index, class_name in enumerate(labels):
        net, mfe, mae, timing, stop = {
            "good": (1.8, 2.7, 0.3, 1.2, 0.0),
            "neutral": (0.1, 1.2, 1.1, 4.5, 0.0),
            "bad": (-1.8, 0.3, 2.7, 8.5, 1.0),
        }[class_name]
        rows.append(
            {
                # Four contemporaneous candidates per cross-section.  The
                # final group crosses into February to exercise month tables.
                "__ts__": pd.Timestamp("2026-01-30T00:00:00Z") + pd.Timedelta(hours=(index // 4) * 36),
                "path_geometry_label": class_name,
                "side": "long" if index % 2 else "short",
                "symbol": "AAA" if index % 3 else "BBB",
                "net_ev_after_1pct_return": net + (index % 2) * 0.01,
                "peak_mfe_12h_atr": mfe + (index % 2) * 0.01,
                "mae_12h_atr": mae - (index % 2) * 0.01,
                "time_to_first_meaningful_mfe": timing + (index % 2) * 0.01,
                "stop_probability": stop,
            }
        )
    frame = pd.DataFrame(rows)
    class_index = {name: index for index, name in enumerate(CLASSES)}
    raw = np.full((len(frame), len(CLASSES)), 0.03, dtype=float)
    for row, label in enumerate(labels):
        raw[row, class_index[label]] = 0.94
    calibrated = raw**1.4
    calibrated /= calibrated.sum(axis=1, keepdims=True)
    return frame, raw, calibrated


def _diagnostics():
    model = PathArchetypeEconomicIC(
        CLASSES,
        config=EconomicICConfig(min_group_rows=2, min_symbol_rows=2),
    ).fit(_train_frame(), provenance={"fold": "train_2025", "oos_outcomes_used_for_fit": False})
    oos, raw, calibrated = _oos_frame()
    return model, oos, raw, calibrated, model.evaluate(oos, {"raw": raw, "calibrated": calibrated})


def test_probability_weighted_train_priors_rank_oos_economic_outcomes() -> None:
    _, _, _, _, result = _diagnostics()
    table = result.global_ic.set_index(["probability_variant", "component"])
    for component in ("net_ev_after_cost", "mfe", "mae", "time_to_realization", "stop_probability", "pooled_economic_utility"):
        assert table.loc[("raw", component), "ic"] > 0.90
    mae = table.loc[("raw", "mae")]
    stop = table.loc[("raw", "stop_probability")]
    assert mae["direction"] == -1.0
    assert bool(mae["higher_is_better"]) is False
    assert stop["direction"] == -1.0
    assert bool(stop["higher_is_better"]) is False


def test_raw_and_calibrated_probability_variants_and_probability_class_ics() -> None:
    _, _, _, _, result = _diagnostics()
    assert set(result.global_ic["probability_variant"]) == {"raw", "calibrated"}
    assert set(result.probability_class_ic["probability_variant"]) == {"raw", "calibrated"}
    assert set(result.probability_class_ic["probability_class"]) == set(CLASSES)
    assert set(result.true_archetype_ic["path_geometry_label"]) == set(CLASSES)
    good_net = result.probability_class_ic.loc[
        (result.probability_class_ic["probability_variant"] == "raw")
        & (result.probability_class_ic["probability_class"] == "good")
        & (result.probability_class_ic["component"] == "net_ev_after_cost")
    ].iloc[0]
    bad_net = result.probability_class_ic.loc[
        (result.probability_class_ic["probability_variant"] == "raw")
        & (result.probability_class_ic["probability_class"] == "bad")
        & (result.probability_class_ic["component"] == "net_ev_after_cost")
    ].iloc[0]
    assert good_net["ic"] > 0.85
    assert bad_net["ic"] < -0.7


def test_timestamp_month_side_and_symbol_neutral_reports_are_emitted() -> None:
    _, _, _, _, result = _diagnostics()
    assert result.timestamp_ic["__timestamp__"].nunique() == 3
    assert result.timestamp_summary["timestamps_with_valid_ic"].min() > 0
    assert set(result.monthly_ic["__month__"]) == {"2026-01", "2026-02"}
    assert set(result.side_ic["side"]) == {"long", "short"}
    assert set(result.symbol_ic["symbol"]) == {"AAA", "BBB"}
    assert result.symbol_neutral_ic["symbol_neutral_weighted_ic"].notna().any()
    assert result.symbol_neutral_ic["interpretation"].eq(
        "diagnostic_only_weighted_aggregation_of_within_symbol_oos_ics"
    ).all()


def test_constant_timestamp_outcome_is_reported_without_silent_drop() -> None:
    model, oos, raw, _, _ = _diagnostics()
    oos.loc[oos.index[:4], "net_ev_after_1pct_return"] = 0.5
    result = model.evaluate(oos, {"raw": raw})
    rows = result.timestamp_ic.loc[
        (result.timestamp_ic["__timestamp__"] == oos.loc[0, "__ts__"])
        & (result.timestamp_ic["component"] == "net_ev_after_cost")
    ]
    assert len(rows) == 1
    assert rows.iloc[0]["status"] == "constant_expected_or_outcome"
    assert np.isnan(rows.iloc[0]["ic"])


def test_train_only_priors_are_invariant_to_oos_outcome_changes() -> None:
    model, oos, raw, calibrated, first = _diagnostics()
    changed = oos.copy()
    for column in (
        "net_ev_after_1pct_return",
        "peak_mfe_12h_atr",
        "mae_12h_atr",
        "time_to_first_meaningful_mfe",
        "stop_probability",
    ):
        changed[column] = changed[column] * -17.0 + 11.0
    second = model.evaluate(changed, {"raw": raw, "calibrated": calibrated})
    pd.testing.assert_frame_equal(first.class_priors, second.class_priors)
    pd.testing.assert_frame_equal(first.class_centroids, second.class_centroids)
    assert first.provenance["oos_outcomes_used_for_fit"] is False
    assert second.quality["oos_outcomes_fit_priors"].eq(False).all()
    assert second.quality["class_labels_consumed_from_oos_for_fit"].eq(False).all()


def test_raw_probability_variant_is_required() -> None:
    model, oos, raw, _, _ = _diagnostics()
    with pytest.raises(ValueError, match="include a raw"):
        model.evaluate(oos, {"calibrated": raw})


def test_persist_keeps_train_only_provenance_and_all_ic_tables(tmp_path) -> None:
    _, _, _, _, result = _diagnostics()
    paths = result.persist(tmp_path)
    assert paths["class_priors"].is_file()
    assert paths["class_centroids"].is_file()
    assert paths["timestamp_summary"].is_file()
    assert paths["manifest"].is_file()
    # Read the JSON directly; ``read_json`` is deliberately not used for the
    # nested manifest object because it normalizes scalar booleans differently.
    import json

    value = json.loads(paths["manifest"].read_text())
    assert value["state_scope"] == "train_only_class_outcome_priors_and_centroids"
    assert value["oos_outcomes_used_only_for_evaluation"] is True
    assert value["provenance"]["oos_outcomes_used_for_fit"] is False
