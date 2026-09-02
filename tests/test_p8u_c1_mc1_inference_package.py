from __future__ import annotations

import numpy as np
import pandas as pd

from extreme_price_movements.inference.p8u_c1_mc1_inference_package import (
    FEATURES,
    build_shift_state,
    fit_package,
    load_package,
    save_package,
)


def _panel(rows: int = 5_200) -> pd.DataFrame:
    rng = np.random.default_rng(1729)
    timestamp = pd.Timestamp("2025-08-01T00:00:00Z") + pd.to_timedelta(
        np.arange(rows) % 192, unit="h"
    )
    frame = pd.DataFrame({
        "candidate_id": [f"C1{i:05d}|long" for i in range(rows)],
        "__decision_ts__": timestamp,
        "policy_path_valid": True,
        "policy_net_bps": rng.normal(80.0, 125.0, rows),
        "policy_label_available_ts": timestamp + pd.Timedelta(hours=12),
        "final_score": np.clip(rng.normal(.7, .15, rows), .0, 1.0),
    })
    for index, field in enumerate(FEATURES):
        frame[field] = rng.normal(.5 + index / 10.0, .2, rows)
    # C1 absence is an allowed explicit model state, never a row removal.
    frame.loc[frame.index[::7], list(FEATURES[6:-1])] = np.nan
    frame.loc[frame.index[::7], "sr_snapshot_available"] = 0.0
    return frame


def test_c1_package_round_trips_with_missing_c1_snapshot_state(tmp_path) -> None:
    train = _panel()
    package = fit_package(
        train,
        family="bcf",
        train_start=pd.Timestamp("2025-02-01T00:00:00Z"),
        train_end_exclusive=pd.Timestamp("2025-08-01T00:00:00Z"),
        held_start=pd.Timestamp("2025-08-01T00:00:00Z"),
        held_end_exclusive=pd.Timestamp("2025-09-01T00:00:00Z"),
        train_months=6,
        source_hashes={"c1_lva": "abc"},
        policy_contract={"target": "canonical_policy_net_bps"},
    )
    static = package.predict_static(train)
    assert np.isfinite(static).all()
    shift = build_shift_state(
        package, train,
        held_start=pd.Timestamp("2025-08-05T00:00:00Z"),
        held_end_exclusive=pd.Timestamp("2025-08-08T00:00:00Z"),
    )
    save_package(package, shift, tmp_path / "package")
    restored = load_package(tmp_path / "package")
    np.testing.assert_allclose(restored.predict_static(train), static, rtol=0.0, atol=0.0)
    assert restored.feature_names == FEATURES
    assert restored.metadata()["source_hashes"]["c1_lva"] == "abc"


def test_c1_package_requires_all_ordered_feature_columns() -> None:
    train = _panel()
    missing = FEATURES[-2]
    with np.testing.assert_raises_regex(KeyError, "fit lacks fields"):
        fit_package(
            train.drop(columns=missing),
            family="current",
            train_start=pd.Timestamp("2025-02-01T00:00:00Z"),
            train_end_exclusive=pd.Timestamp("2025-08-01T00:00:00Z"),
            held_start=pd.Timestamp("2025-08-01T00:00:00Z"),
            held_end_exclusive=pd.Timestamp("2025-09-01T00:00:00Z"),
            train_months=6,
            source_hashes={}, policy_contract={},
        )
