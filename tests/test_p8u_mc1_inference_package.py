from __future__ import annotations

from dataclasses import replace

import joblib

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.inference.p8u_mc1_inference_package import (
    FEATURES,
    apply_shift,
    build_shift_state,
    fit_package,
    load_package,
    save_package,
)


def _panel(rows: int = 5_100) -> pd.DataFrame:
    rng = np.random.default_rng(1729)
    timestamp = pd.Timestamp("2025-08-01T00:00:00Z") + pd.to_timedelta(np.arange(rows) % 192, unit="h")
    frame = pd.DataFrame({
        "candidate_id": [f"X{i:05d}|long" for i in range(rows)],
        "__decision_ts__": timestamp,
        "policy_path_valid": True,
        "policy_net_bps": rng.normal(80.0, 125.0, rows),
        "policy_label_available_ts": timestamp + pd.Timedelta(hours=12),
    })
    for idx, field in enumerate(FEATURES):
        frame[field] = rng.normal(0.5 + idx / 10.0, 0.2, rows)
    frame["final_score"] = np.clip(frame["final_score"], 0.0, 1.0)
    return frame


def test_six_month_package_round_trips_and_scores_identically(tmp_path):
    train = _panel()
    package = fit_package(
        train,
        family="bcf",
        train_start=pd.Timestamp("2025-02-01T00:00:00Z"),
        train_end_exclusive=pd.Timestamp("2025-08-01T00:00:00Z"),
        held_start=pd.Timestamp("2025-08-01T00:00:00Z"),
        held_end_exclusive=pd.Timestamp("2025-09-01T00:00:00Z"),
        train_months=6,
        source_hashes={"test": "abc"},
        policy_contract={"test": "policy"},
    )
    shift = build_shift_state(
        package, train,
        held_start=pd.Timestamp("2025-08-05T00:00:00Z"),
        held_end_exclusive=pd.Timestamp("2025-08-08T00:00:00Z"),
    )
    save_package(package, shift, tmp_path / "package")
    restored = load_package(tmp_path / "package")
    np.testing.assert_allclose(restored.predict_static(train), package.predict_static(train), rtol=0.0, atol=0.0)
    assert restored.feature_names == FEATURES
    assert restored.metadata()["train_months"] == 6


def test_shift_excludes_not_yet_resolved_outcomes():
    train = _panel()
    package = fit_package(
        train,
        family="current",
        train_start=pd.Timestamp("2025-02-01T00:00:00Z"),
        train_end_exclusive=pd.Timestamp("2025-08-01T00:00:00Z"),
        held_start=pd.Timestamp("2025-08-01T00:00:00Z"),
        held_end_exclusive=pd.Timestamp("2025-09-01T00:00:00Z"),
        train_months=6,
        source_hashes={}, policy_contract={},
    )
    day = pd.Timestamp("2025-08-08T00:00:00Z")
    history = train.copy()
    history.loc[history.index[:100], "__decision_ts__"] = day - pd.Timedelta(hours=1)
    history.loc[history.index[:100], "policy_label_available_ts"] = day + pd.Timedelta(hours=1)
    history.loc[history.index[:100], "policy_net_bps"] = 1_000_000.0
    state = build_shift_state(package, history, held_start=day, held_end_exclusive=day + pd.Timedelta(days=1))
    assert pd.isna(state.loc[0, "max_policy_label_available_ts"]) or state.loc[0, "max_policy_label_available_ts"] < day
    applied = apply_shift(np.array([1.0]), [day], state)
    assert np.isfinite(applied).all()


def test_package_rejects_non_six_month_contract():
    with pytest.raises(ValueError, match="exactly six"):
        fit_package(
            _panel(), family="bcf",
            train_start=pd.Timestamp("2025-02-01T00:00:00Z"),
            train_end_exclusive=pd.Timestamp("2025-08-01T00:00:00Z"),
            held_start=pd.Timestamp("2025-08-01T00:00:00Z"),
            held_end_exclusive=pd.Timestamp("2025-09-01T00:00:00Z"),
            train_months=3, source_hashes={}, policy_contract={},
        )


def test_package_rejects_noncontiguous_or_noncalendar_six_month_window(tmp_path):
    train = _panel()
    common = {
        "family": "bcf",
        "train_start": pd.Timestamp("2025-03-01T00:00:00Z"),
        "train_end_exclusive": pd.Timestamp("2025-08-01T00:00:00Z"),
        "held_start": pd.Timestamp("2025-08-01T00:00:00Z"),
        "held_end_exclusive": pd.Timestamp("2025-09-01T00:00:00Z"),
        "train_months": 6,
        "source_hashes": {},
        "policy_contract": {},
    }
    with pytest.raises(ValueError, match="exactly six complete"):
        fit_package(train, **common)
    common["train_start"] = pd.Timestamp("2025-02-01T12:00:00Z")
    with pytest.raises(ValueError, match="calendar-month boundary"):
        fit_package(train, **common)

    package = fit_package(
        train,
        family="bcf",
        train_start=pd.Timestamp("2025-02-01T00:00:00Z"),
        train_end_exclusive=pd.Timestamp("2025-08-01T00:00:00Z"),
        held_start=pd.Timestamp("2025-08-01T00:00:00Z"),
        held_end_exclusive=pd.Timestamp("2025-09-01T00:00:00Z"),
        train_months=6,
        source_hashes={},
        policy_contract={},
    )
    bad_dir = tmp_path / "bad_package"
    bad_dir.mkdir()
    joblib.dump(replace(package, train_start="2025-03-01T00:00:00+00:00"), bad_dir / "package.joblib")
    with pytest.raises(ValueError, match="exactly six complete"):
        load_package(bad_dir)
