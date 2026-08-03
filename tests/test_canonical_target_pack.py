from __future__ import annotations

import json

import pandas as pd

from extreme_price_movements.canonical_target_pack import (
    METADATA_COLUMNS,
    _metadata_dictionary,
    augment_support_report,
)


def test_metadata_dictionary_is_unique_and_forbidden_as_input() -> None:
    dictionary = _metadata_dictionary()
    assert len(dictionary) == 20
    assert dictionary.label_name.is_unique
    assert not dictionary.model_input_allowed.any()
    assert set(dictionary.label_name) == set(METADATA_COLUMNS)


def test_support_report_appends_metadata_rows() -> None:
    labels = pd.DataFrame({
        "candidate_id": ["a", "b"],
        "decision_ts": pd.to_datetime(["2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z"]),
        "side": ["long", "short"],
        **{column: [0, 1] for column in METADATA_COLUMNS},
    })
    source = pd.DataFrame(columns=["surface", "month", "side", "label_name", "rows", "non_null_rows", "mean", "std", "p05", "p50", "p95"])
    out = augment_support_report(source, labels)
    assert len(out) == len(METADATA_COLUMNS) * 2
    assert set(out.surface) == {"supportive_metadata"}
    assert out.label_name.isin(METADATA_COLUMNS).all()
