from __future__ import annotations

import numpy as np
import pandas as pd

from extreme_price_movements.lgbm_pipeline import (
    _ensure_feature_family_coverage,
    _redundancy_cluster_filter,
)


def test_feature_family_coverage_adds_evidenced_representatives() -> None:
    candidates = ["ret_1h", "mkt_breadth_1h", "oi_chg_4h", "leaf_support_count"]
    scores = {
        "ret_1h": 0.8,
        "mkt_breadth_1h": 0.5,
        "oi_chg_4h": 0.4,
        "leaf_support_count": 0.3,
    }

    selected = _ensure_feature_family_coverage(["ret_1h"], candidates, scores)

    assert "mkt_breadth_1h" in selected
    assert "oi_chg_4h" in selected
    assert "leaf_support_count" in selected


def test_redundancy_filter_accepts_archetype_labels() -> None:
    rng = np.random.default_rng(3)
    n = 800
    base = rng.normal(size=n).astype(np.float32)
    frame = pd.DataFrame(
        {
            "ret_fast": base,
            "ret_duplicate": base + rng.normal(0, 0.001, n),
            "mkt_breadth_1h": rng.normal(size=n),
        }
    )
    labels = np.where(np.arange(n) < n // 2, "long_a", "short_b")
    selected = _redundancy_cluster_filter(
        frame,
        list(frame.columns),
        {"ret_fast": 1.0, "ret_duplicate": 0.5, "mkt_breadth_1h": 0.8},
        random_state=7,
        archetype_labels=labels,
    )

    assert "ret_fast" in selected
    assert "mkt_breadth_1h" in selected


def test_correlation_first_filter_respects_floor_and_protected_features() -> None:
    rng = np.random.default_rng(17)
    rows = 900
    base = rng.normal(size=(rows, 10)).astype(np.float32)
    data: dict[str, np.ndarray] = {}
    for group in range(10):
        for duplicate in range(40):
            data[f"f_{group}_{duplicate}"] = (
                base[:, group] + rng.normal(0.0, 1e-4, rows)
            ).astype(np.float32)
    frame = pd.DataFrame(data)
    protected = "f_0_39"
    selected = _redundancy_cluster_filter(
        frame,
        list(frame.columns),
        {},
        random_state=19,
        min_retained=300,
        representative_policy="availability_variance",
        protected_features=[protected],
    )

    assert len(selected) >= 300
    assert len(selected) <= len(frame.columns)
    assert protected in selected


def test_correlation_first_representative_is_target_independent() -> None:
    rng = np.random.default_rng(23)
    base = rng.normal(size=800).astype(np.float32)
    frame = pd.DataFrame(
        {
            "a": base,
            "b": base + rng.normal(0.0, 1e-4, len(base)),
            "c": rng.normal(size=len(base)),
        }
    )
    first = _redundancy_cluster_filter(
        frame,
        list(frame.columns),
        {"a": -100.0, "b": 100.0},
        random_state=29,
        representative_policy="availability_variance",
    )
    second = _redundancy_cluster_filter(
        frame,
        list(frame.columns),
        {"a": 100.0, "b": -100.0},
        random_state=29,
        representative_policy="availability_variance",
    )
    assert first == second
