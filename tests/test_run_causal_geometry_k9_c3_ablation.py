from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts" / "run_causal_geometry_k9_c3_ablation.py"
SPEC = importlib.util.spec_from_file_location("c3_geometry_runner", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_c3_windows_use_a_three_month_burn_in_and_post_burn_in_meta_start() -> None:
    cutoff = pd.Timestamp("2026-04-01", tz="UTC")
    frozen = MODULE._geometry_window("c3_frozen", cutoff)
    quarterly = MODULE._geometry_window("c3_quarterly", cutoff)
    rolling = MODULE._geometry_window("c3_rolling", cutoff)

    assert frozen == (
        pd.Timestamp("2025-01-01", tz="UTC"),
        pd.Timestamp("2025-04-01", tz="UTC"),
        pd.Timestamp("2025-04-01", tz="UTC"),
    )
    assert quarterly == (
        pd.Timestamp("2025-04-01", tz="UTC"),
        pd.Timestamp("2025-07-01", tz="UTC"),
        pd.Timestamp("2025-07-01", tz="UTC"),
    )
    assert rolling == quarterly


def test_raw_k9_bundle_is_base_and_outcome_independent() -> None:
    fields = [f"f{index:03d}" for index in range(120)]
    rows = 90
    frame = pd.DataFrame(np.arange(rows * len(fields), dtype=np.float32).reshape(rows, len(fields)), columns=fields)
    frame["__ts__"] = pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC")
    frame["__symbol__"] = "BTC/USD:USD"
    frame["candidate_id"] = [f"id-{index}" for index in range(rows)]

    bundle, audit = MODULE._fit_raw_k9(
        frame,
        fields,
        bundle_id="unit",
        fit_start=pd.Timestamp("2025-01-01", tz="UTC"),
        fit_end=pd.Timestamp("2025-02-01", tz="UTC"),
        source_kind="unit_test_raw_only",
        previous=None,
    )
    transformed = bundle.transform(frame)

    assert audit["fit_uses_outcomes"] is False
    assert audit["base_independent"] is True
    assert transformed.shape == (rows, 3 * MODULE.K + 9)
    assert np.isfinite(transformed.to_numpy()).all()


def test_raw_k9_membership_temperature_is_frozen_from_fit_population() -> None:
    fields = ("f0", "f1")
    fit = pd.DataFrame(
        {
            "__decision_ts__": pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC"),
            "candidate_id": [f"fit-{index}" for index in range(200)],
            "f0": np.linspace(-2.0, 2.0, 200),
            "f1": np.sin(np.linspace(0.0, 5.0, 200)),
        }
    )
    bundle, audit = MODULE._fit_raw_k9(
        fit,
        fields,
        bundle_id="frozen-temperature-test",
        fit_start=pd.Timestamp("2024-01-01", tz="UTC"),
        fit_end=pd.Timestamp("2024-02-01", tz="UTC"),
        source_kind="test",
        previous=None,
    )
    score = fit.iloc[:20].copy()
    first = bundle.transform(score)
    extreme = score.copy()
    extreme["f0"] = 1_000.0
    extreme["f1"] = -1_000.0
    combined = bundle.transform(pd.concat([score, extreme], ignore_index=True)).iloc[: len(score)]

    legacy = [
        field for field in first
        if not field.startswith("geometry_")
    ]
    np.testing.assert_allclose(
        first[legacy].to_numpy(), combined[legacy].to_numpy(), rtol=0.0, atol=1e-7,
    )
    assert not np.allclose(
        first["geometry_mahalanobis_train"],
        combined["geometry_mahalanobis_train"],
    )
    assert audit["membership_temperature_source"] == "geometry_fit_population_only"


def test_structural_break_is_invariant_to_later_timestamps() -> None:
    fields = ("f0", "f1")
    fit = pd.DataFrame(
        {
            "__decision_ts__": pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC"),
            "candidate_id": [f"fit-{index}" for index in range(200)],
            "f0": np.linspace(-2.0, 2.0, 200),
            "f1": np.sin(np.linspace(0.0, 5.0, 200)),
        }
    )
    bundle, _ = MODULE._fit_raw_k9(
        fit, fields, bundle_id="future-invariance",
        fit_start=pd.Timestamp("2024-01-01", tz="UTC"),
        fit_end=pd.Timestamp("2024-02-01", tz="UTC"),
        source_kind="test", previous=None,
    )
    early = fit.iloc[:20].copy()
    future = fit.iloc[20:40].copy()
    future["__decision_ts__"] += pd.Timedelta(days=100)
    first = bundle.transform(early)
    combined = bundle.transform(pd.concat([early, future], ignore_index=True)).iloc[: len(early)]
    structural = [column for column in first if column.startswith("geometry_")]
    np.testing.assert_allclose(first[structural], combined[structural], rtol=0.0, atol=1e-7)


def test_raw_k9_temperature_scale_sharpens_membership_without_refitting_centres() -> None:
    fields = ("f0", "f1")
    fit = pd.DataFrame(
        {
            "__decision_ts__": pd.date_range("2024-01-01", periods=400, freq="h", tz="UTC"),
            "candidate_id": [f"fit-{index}" for index in range(400)],
            "f0": np.linspace(-3.0, 3.0, 400),
            "f1": np.cos(np.linspace(0.0, 8.0, 400)),
        }
    )
    bundle, audit = MODULE._fit_raw_k9(
        fit,
        fields,
        bundle_id="temperature-scale-test",
        fit_start=pd.Timestamp("2024-01-01", tz="UTC"),
        fit_end=pd.Timestamp("2024-02-01", tz="UTC"),
        source_kind="test",
        previous=None,
        temperature_scale=0.50,
    )
    sharpened = bundle.transform(fit.iloc[:100])
    bundle.temperature_scale = 1.0
    control = bundle.transform(fit.iloc[:100])

    assert sharpened["k9_entropy"].mean() < control["k9_entropy"].mean()
    assert sharpened["k9_top2_margin"].mean() > control["k9_top2_margin"].mean()
    assert audit["membership_temperature_scale"] == 0.50
    assert audit["effective_membership_temperature"] == 0.50 * audit["membership_temperature"]


def test_base_coupled_k9_uses_the_same_leaf_reference_and_marks_in_sample_risk() -> None:
    class FakeLeafModel:
        def predict(self, values: np.ndarray, *, pred_leaf: bool = False) -> np.ndarray:
            assert pred_leaf
            base = (np.arange(len(values), dtype=np.int32) % 4)[:, None]
            return (base + np.arange(64, dtype=np.int32)[None, :]) % 8

    fields = ("f0", "f1")
    train_leaves = np.tile(np.arange(64, dtype=np.int32), (96, 1)) % 8
    reference = MODULE.LeafReference(
        fields=fields,
        medians=np.zeros(2, dtype=np.float32),
        model=FakeLeafModel(),
        train_leaves=train_leaves,
        support_counts=tuple(np.bincount(train_leaves[:, tree]).astype(np.float32) for tree in range(64)),
        train_rows=len(train_leaves),
    )
    frame = pd.DataFrame({
        "f0": np.linspace(-1.0, 1.0, 96),
        "f1": np.linspace(1.0, -1.0, 96),
        "__decision_ts__": pd.date_range("2025-01-01", periods=96, freq="h", tz="UTC"),
        "candidate_id": [f"id-{index}" for index in range(96)],
    })
    bundle, audit = MODULE._fit_base_coupled_k9(
        reference,
        frame,
        bundle_id="unit-base-coupled",
        fit_start=pd.Timestamp("2025-01-01", tz="UTC"),
        fit_end=pd.Timestamp("2025-02-01", tz="UTC"),
        lookback_months=3,
    )
    transformed = bundle.transform(frame)

    assert bundle.leaf_reference is reference
    assert audit["fit_uses_outcomes"] is True
    assert audit["same_leaf_reference_for_k9_and_state"] is True
    assert audit["in_sample_meta_rows_allowed"] is True
    assert transformed.shape == (len(frame), 3 * MODULE.K + 3)
    assert np.isfinite(transformed.to_numpy()).all()


def test_base_coupled_lookback_grid_is_bounded() -> None:
    assert MODULE._base_coupled_lookback_months("basecoupled_in_sample_3m") == 3
    assert MODULE._base_coupled_lookback_months("basecoupled_in_sample_6m") == 6
    assert MODULE._base_coupled_lookback_months("basecoupled_in_sample_9m") == 9
    assert MODULE._base_coupled_lookback_months("c3_rolling") is None
