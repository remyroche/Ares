from __future__ import annotations

import numpy as np
import pandas as pd

from extreme_price_movements.base_leaf_representations import (
    FrozenBaseLeafDictionary,
    cap_support_diverse,
    strict_dictionary_split,
)
from extreme_price_movements.performance_regimes.correctness_leaf_regimes import LeafRule


def _frame() -> pd.DataFrame:
    decision = pd.date_range("2024-01-01", periods=40, freq="h", tz="UTC")
    return pd.DataFrame({
        "candidate_id": [f"c{i}" for i in range(40)],
        "__ts__": decision - pd.Timedelta(hours=1),
        "__symbol__": "X", "side_name": "long", "decision_ts": decision,
        "label_available_ts": decision + pd.Timedelta(hours=12),
        "r3_class": np.resize(np.array([0, 1, 2]), 40), "x": np.linspace(-1, 1, 40),
    })


def test_dictionary_labels_resolve_before_later_base_rows() -> None:
    dictionary, later = strict_dictionary_split(_frame(), dictionary_fraction=.45)
    assert dictionary.label_available_ts.max() < later.decision_ts.min()
    assert len(dictionary) and len(later)


def test_frozen_membership_does_not_read_outcome_columns() -> None:
    raw = _frame().iloc[14:].copy()
    frozen = FrozenBaseLeafDictionary(
        side="long", fold_id=2, target_name="row", features=("x",),
        median={"x": 0.0}, iqr={"x": 1.0},
        clusters=((LeafRule("t0_l1", (("x", 1, 0.0),), .5, .5), LeafRule("t1_l1", (("x", 1, -.2),), .2, .2)),),
        rule_similarity=pd.DataFrame(), dictionary_rows=1000,
        dictionary_max_label_available_utc="2024-01-01T00:00:00+00:00",
        applied_from_decision_utc="2024-01-02T00:00:00+00:00",
    )
    first, _ = frozen.apply(raw)
    raw["r3_class"] = 2 - raw["r3_class"]
    raw["exact_net_bps"] = 999_999.0
    second, _ = frozen.apply(raw)
    feature = [column for column in first if column.startswith("baseleaf__")][0]
    np.testing.assert_allclose(first[feature], second[feature])


def test_sparse_support_is_reserved_in_the_cap() -> None:
    ranking = pd.DataFrame({
        "feature": [f"sparse{i}" for i in range(5)] + [f"mid{i}" for i in range(7)] + [f"broad{i}" for i in range(15)],
        "active_share": [.06] * 5 + [.15] * 7 + [.30] * 15,
        "min_block_mda": np.arange(27, 0, -1, dtype=float),
    })
    selected = cap_support_diverse(ranking)
    assert sum(item.startswith("sparse") for item in selected) == 3
    assert sum(item.startswith("mid") for item in selected) == 5
    assert len(selected) == 20
