from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ablate_strict_r3_cell_day_bayesian_ev_mapping.py"
SPEC = importlib.util.spec_from_file_location("cell_day_bayes", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_canonical_cell_day_window_is_28_days() -> None:
    assert MODULE.DEFAULT_WINDOW_DAYS == 28


def test_cell_day_table_gives_each_day_one_observation() -> None:
    frame = pd.DataFrame({
        "__day__": pd.to_datetime(["2026-01-01"] * 100 + ["2026-01-02"], utc=True),
        "__cell__": [0] * 101,
        "policy_net_bps": [100.0] * 100 + [-100.0],
        "candidate_id": [f"row-{i}" for i in range(101)],
    })
    table = MODULE._cell_day_table(frame)
    curve, support = MODULE._equal_day_curve(table, trim=0.0)
    assert support[0] == 2
    assert np.isclose(curve[0], 0.0)


def test_cell_day_trim_is_symmetric() -> None:
    values = np.arange(20, dtype=float)
    retained = MODULE._trim_values(values, 0.10)
    np.testing.assert_array_equal(retained, np.arange(2, 18, dtype=float))


def test_reactive_trim_keeps_recent_bottom_day() -> None:
    snapshot = pd.Timestamp("2026-02-01T00:00:00Z")
    days = pd.date_range("2026-01-12", periods=20, freq="D", tz="UTC")
    values = np.arange(20, dtype=float)
    # Make the most recent resolved day the most adverse observation.  It must
    # remain even though an ordinary symmetric trim would remove it.
    values[-1] = -1000.0
    table = pd.DataFrame({
        "__day__": days, "__cell__": 0,
        "cell_day_ev_bps": values, "cell_day_trades": 100,
    })
    curve, support = MODULE._reactive_equal_day_curve(
        table, trim=0.10, snapshot=snapshot,
    )
    assert support[0] == 16
    assert curve[0] < 0.0


def test_bayesian_probability_increases_with_positive_cell_days() -> None:
    days = pd.date_range("2026-01-01", periods=10, freq="D", tz="UTC")
    table = pd.DataFrame({
        "__day__": days,
        "__cell__": [19] * 10,
        "cell_day_ev_bps": np.linspace(50.0, 150.0, 10),
        "cell_day_trades": 100,
    })
    curve, probability, support = MODULE._bayesian_curve(
        table, model_prior=np.zeros(20), prior_days=7.0, trim=0.0,
    )
    assert support[19] == 10
    assert curve[19] > 0.0
    assert probability[19] > 0.90


def test_reference_bins_are_bounded() -> None:
    reference = np.linspace(0.0, 1.0, 1000)
    bins = MODULE._reference_bins(reference, np.array([-1.0, 0.5, 2.0]))
    assert bins.tolist() == [0, 10, 19]


def test_materializer_emits_authoritative_portfolio_provenance_contract() -> None:
    text = SCRIPT.read_text()
    for field in (
        "conversion_bundle_sha256", "upstream_bundle_sha256",
        "geometry_bundle_sha256", "ev_score_family_id", "stack_is_prequential",
        "causal_21d_side_expected_net_bps",
        "causal_21d_side_admitted_ge_50bps",
        "causal_21d_side_mapping_status",
        "cell_day_trim_15pct__fixed_score_cell",
        "cell_day_trim_15pct__retained_day_support",
    ):
        assert field in text
    assert "CELL_DAY_TRIM_15_CALIBRATION_MODE" in text
    assert '"maximum_label_available_ts": maximum_label_available' in text
    assert "maximum_label_available < day" in text
    assert 'canonical_audit[\n        "strictly_prior_resolved"' in text
