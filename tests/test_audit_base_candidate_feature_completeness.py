from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_base_candidate_feature_completeness import (
    _combine_group_summaries,
    _feature_sources,
    _handoff_months,
    _read_handoff_month,
    load_base_selected_feature_contract,
    summarize_completeness,
)


def test_summarize_completeness_counts_nan_and_infinity_as_violations() -> None:
    keys = pd.DataFrame(
        {
            "__ts__": pd.date_range("2026-07-01", periods=3, freq="h", tz="UTC"),
            "__symbol__": ["AAA", "AAA", "BBB"],
            "side": [1, -1, 1],
        }
    )
    matrix = pd.DataFrame(
        {
            "raw": [1.0, np.nan, 3.0],
            "generated": [0.1, np.inf, 0.3],
            "side": [1.0, -1.0, 1.0],
        }
    )

    summary, missing, details = summarize_completeness(
        keys, matrix, ["raw", "generated", "side"]
    )

    assert summary == {
        "rows": 3,
        "complete_rows": 2,
        "incomplete_rows": 1,
        "joint_complete_fraction": pytest.approx(2.0 / 3.0),
    }
    assert missing == {"raw": 1, "generated": 1, "side": 0}
    assert details["required_input_complete"].tolist() == [True, False, True]


def test_group_summaries_combine_month_side_chunks_without_averaging_fractions() -> None:
    rows = [
        {"month": "2026-01", "side": "long", "rows": 2, "complete_rows": 1, "incomplete_rows": 1},
        {"month": "2026-01", "side": "long", "rows": 8, "complete_rows": 8, "incomplete_rows": 0},
        {"month": "2026-01", "side": "short", "rows": 5, "complete_rows": 0, "incomplete_rows": 5},
    ]

    monthly = _combine_group_summaries(rows, ["month"])
    sides = _combine_group_summaries(rows, ["side"])

    assert monthly == [
        {
            "month": "2026-01",
            "rows": 15,
            "complete_rows": 9,
            "incomplete_rows": 6,
            "joint_complete_fraction": pytest.approx(0.6),
        }
    ]
    assert {row["side"]: row["joint_complete_fraction"] for row in sides} == {
        "long": pytest.approx(0.9),
        "short": pytest.approx(0.0),
    }


def test_feature_sources_rejects_aegmm_columns_not_declared_by_sidecar() -> None:
    raw, generated = _feature_sources(
        ["raw", "gmm_prob_0", "side"], ["gmm_prob_0"]
    )
    assert raw == ["raw"]
    assert generated == ["gmm_prob_0"]

    with pytest.raises(ValueError, match="missing from frozen sidecar"):
        _feature_sources(["raw", "gmm_prob_1"], ["gmm_prob_0"])


def test_contract_loader_requires_one_ordered_shared_150_column_contract(tmp_path: Path) -> None:
    models = tmp_path / "models"
    features = [f"feature_{index:03d}" for index in range(150)]
    for fold in ("fold_a", "fold_b"):
        target = models / fold
        target.mkdir(parents=True)
        (target / "columns.json").write_text(
            json.dumps({"feature_names": features}), encoding="utf-8"
        )

    loaded, paths = load_base_selected_feature_contract(tmp_path)
    assert loaded == features
    assert len(paths) == 2

    (models / "fold_b" / "columns.json").write_text(
        json.dumps({"feature_names": list(reversed(features))}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="differ in ordered content"):
        load_base_selected_feature_contract(tmp_path)


def test_handoff_month_reader_uses_persisted_top30_and_normalizes_side(tmp_path: Path) -> None:
    handoff = tmp_path / "handoff.parquet"
    pd.DataFrame(
        {
            "__ts__": pd.to_datetime(
                ["2026-01-31T23:00:00Z", "2026-02-01T00:00:00Z", "2026-02-01T01:00:00Z"],
                utc=True,
            ),
            "__symbol__": ["AAA", "AAA", "BBB"],
            "side_name": ["long", "short", "long"],
            "selected_top30": [True, True, False],
        }
    ).to_parquet(handoff, index=False)

    months = _handoff_months(handoff, side_column="side_name", side_filter="all")
    assert [month.strftime("%Y-%m") for month in months] == ["2026-01", "2026-02"]
    selected = _read_handoff_month(
        handoff,
        month=months[1],
        side_column="side_name",
        side_filter="all",
    )

    assert selected[["__symbol__", "side"]].to_dict(orient="records") == [
        {"__symbol__": "AAA", "side": -1}
    ]
