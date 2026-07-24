from __future__ import annotations

import pandas as pd
import pytest

from extreme_price_movements.path_archetype_support import (
    FAST_REALIZATION_WINNER,
    MERGED_PATH_ARCHETYPE_CLASSES,
    PathArchetypeSupportConfig,
    merge_fast_realization_winner,
    validate_path_archetype_support,
)


def _frame(labels: list[str], *, timestamps: list[object] | None = None, sides: list[str] | None = None) -> pd.DataFrame:
    count = len(labels)
    return pd.DataFrame(
        {
            "path_geometry_label": labels,
            "__ts__": timestamps or pd.date_range("2026-01-01", periods=count, freq="h", tz="UTC"),
            "side": sides or ["long"] * count,
            "preserved": list(range(count)),
        }
    )


def test_exact_fast_merge_preserves_all_other_values_and_columns() -> None:
    frame = _frame(["fast_clean_winner", "fast_winner_early_drawdown", "slow_grinder"])
    merged = merge_fast_realization_winner(frame)
    assert merged["path_geometry_label"].tolist() == [FAST_REALIZATION_WINNER, FAST_REALIZATION_WINNER, "slow_grinder"]
    assert merged["preserved"].tolist() == frame["preserved"].tolist()
    assert frame["path_geometry_label"].tolist()[0] == "fast_clean_winner"


def test_global_floor_rejects_missing_explicit_classes() -> None:
    result = validate_path_archetype_support(_frame(["slow_grinder"] * 100))
    assert result.recommended_action == "reject_geometry"
    assert set(MERGED_PATH_ARCHETYPE_CLASSES).difference(result.global_support.loc[result.global_support.rows.gt(0), "path_geometry_label"])
    assert (result.violations["scope"] == "global").any()


def test_month_side_floor_rejects_sparse_class_even_when_global_floor_passes() -> None:
    classes = list(MERGED_PATH_ARCHETYPE_CLASSES)
    labels = [classes[0]] * 100 + [classes[1]] * 100 + [classes[2]] * 100 + [classes[3]] * 100 + [classes[4]] * 100 + [classes[5]] * 100 + [classes[6]] * 100
    frame = _frame(labels, timestamps=[pd.Timestamp("2026-01-01", tz="UTC")] * 699 + [pd.Timestamp("2026-02-01", tz="UTC")], sides=["long"] * 700)
    frame.loc[699, "path_geometry_label"] = classes[6]
    result = validate_path_archetype_support(frame, PathArchetypeSupportConfig(min_global_class_share=0.001, min_month_side_class_share=0.005))
    assert result.recommended_action == "reject_geometry"
    assert ((result.violations["scope"] == "month_side") & (result.violations["__month__"] == "2026-02")).any()


def test_utc_timestamp_conversion_groups_calendar_months_in_utc() -> None:
    labels = [MERGED_PATH_ARCHETYPE_CLASSES[0]] * 100
    timestamps = ["2026-01-31T23:30:00-02:00"] * 100
    result = validate_path_archetype_support(_frame(labels, timestamps=timestamps))
    assert set(result.month_support["__month__"]) == {"2026-02"}


def test_tiny_denominator_requires_explicit_opt_in_and_is_reported() -> None:
    frame = _frame([MERGED_PATH_ARCHETYPE_CLASSES[0]], timestamps=[pd.Timestamp("2026-02-01", tz="UTC")])
    default_result = validate_path_archetype_support(frame)
    exempted_result = validate_path_archetype_support(frame, PathArchetypeSupportConfig(min_month_side_denominator=10))
    assert (default_result.violations["scope"] == "month_side").any()
    assert not exempted_result.exemptions.empty
    assert set(exempted_result.exemptions["status"]) == {"explicit_tiny_denominator_exemption"}


def test_legacy_labels_are_rejected_without_silent_repair() -> None:
    frame = _frame(["fast_clean_winner"] * 100)
    result = validate_path_archetype_support(frame)
    assert result.recommended_action == "reject_geometry"
    assert "fast_clean_winner" in set(frame["path_geometry_label"])
    assert "legacy_fast_labels_not_merged" in set(result.violations["status"])


def test_classes_must_be_explicit_merged_taxonomy() -> None:
    with pytest.raises(ValueError, match="merged fast_realization"):
        validate_path_archetype_support(_frame(["slow_grinder"]), PathArchetypeSupportConfig(classes=("fast_clean_winner",)))
