from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.run_base_side_target_geometry_hpo import (
    PathPrimitives,
    SideTargetGeometry,
    _load_params,
    _load_source_payload,
    _topk_net_objective,
    build_internal_chronological_folds,
    build_path_primitives,
    continuous_target,
    geometry_outcomes,
    short_late_continuation_pressure,
)


def _path_metrics() -> pd.DataFrame:
    """Small exact-grid fixture: TP first, SL first, then timeout."""

    values: dict[str, list[float]] = {
        "__barrier_pct__": [0.01, 0.01, 0.01],
        "__y_ret__": [-0.02, -0.03, -0.02],
        "__first_touch_round_trip_cost__": [0.01, 0.01, 0.01],
        "__first_touch_effective_tp_abs__": [0.01, 0.01, 0.01],
        "__first_touch_effective_sl_abs__": [0.01, 0.01, 0.01],
        "__is_timeout__": [0.0, 0.0, 1.0],
    }
    for key in ("05", "075", "1", "125", "15"):
        values[f"__bars_to_mfe_{key}r__"] = [-1.0, -1.0, -1.0]
    for key in ("05", "075", "1", "15"):
        values[f"__bars_to_mae_{key}r__"] = [-1.0, -1.0, -1.0]
    values["__bars_to_mfe_05r__"][0] = 2.0
    values["__bars_to_mfe_075r__"][0] = 4.0
    values["__bars_to_mfe_075r__"][1] = 4.0
    values["__bars_to_mae_05r__"][1] = 2.0
    return pd.DataFrame(values)


def test_exact_supported_geometry_replays_actual_gross_first_touch_outcomes() -> None:
    primitives = build_path_primitives(_path_metrics())
    tight = SideTargetGeometry(0.50, 0.50, 8, 0.0, 0.01, 0.5, 0.2, 16.0, 0.2)
    wider_tp = SideTargetGeometry(0.75, 0.50, 3, 0.0, 0.01, 0.5, 0.2, 16.0, 0.2)

    gross, tp, sl, timeout, bars = geometry_outcomes(primitives, tight)
    wider_gross, wider_tp, wider_sl, wider_timeout, _ = geometry_outcomes(primitives, wider_tp)

    # Row 0 hits +0.5R before any stop under the tight geometry, but becomes a
    # true terminal-path timeout under +0.75R. Row 1 stops first. Row 2 is a
    # timeout in both cases and uses __y_ret__ + one stored cost.
    assert np.allclose(gross, np.asarray([0.005, -0.005, -0.01], dtype=np.float32))
    assert np.allclose(gross - primitives.round_trip_cost, np.asarray([-0.005, -0.015, -0.02], dtype=np.float32))
    assert np.array_equal(tp, np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
    assert np.array_equal(sl, np.asarray([0.0, 1.0, 0.0], dtype=np.float32))
    assert np.array_equal(timeout, np.asarray([0.0, 0.0, 1.0], dtype=np.float32))
    assert np.array_equal(bars, np.asarray([2.0, 2.0, 8.0], dtype=np.float32))
    assert wider_gross[0] == pytest.approx(-0.01)
    assert wider_tp[0] == 0.0 and wider_sl[0] == 0.0 and wider_timeout[0] == 1.0


def test_continuous_target_remains_soft_but_is_geometry_sensitive() -> None:
    primitives = build_path_primitives(_path_metrics())
    tight = SideTargetGeometry(0.50, 0.50, 8, 0.0, 0.01, 0.5, 0.2, 16.0, 0.2)
    strict = SideTargetGeometry(0.75, 0.50, 8, 0.0, 0.01, 1.5, 1.0, 4.0, 1.0)

    tight_target = continuous_target(primitives, tight)
    strict_target = continuous_target(primitives, strict)

    assert tight_target.dtype == np.float32
    assert np.all((tight_target >= 0.0) & (tight_target <= 1.0))
    assert not np.allclose(tight_target, strict_target)
    assert tight_target[0] > tight_target[1]


def test_stage_c_objective_is_global_and_subtracts_cost_once() -> None:
    score = np.asarray([0.9, 0.8, 0.7, 0.6], dtype=np.float32)
    gross = np.asarray([0.03, 0.02, 0.00, -0.01], dtype=np.float32)
    cost = np.full(4, 0.01, dtype=np.float32)

    objective, metrics = _topk_net_objective(
        score,
        gross,
        cost,
        np.arange(4),
        top_weights={0.10: 0.50, 0.20: 0.30, 0.30: 0.20},
    )

    assert metrics["net_top10"] == pytest.approx(0.02)
    assert metrics["net_top20"] == pytest.approx(0.02)
    assert metrics["net_top30"] == pytest.approx(0.015)
    assert objective == pytest.approx(0.019)


def test_exact_geometry_rejects_missing_or_unsupported_primitives() -> None:
    with pytest.raises(ValueError, match="__bars_to_mfe_15r__"):
        build_path_primitives(_path_metrics().drop(columns="__bars_to_mfe_15r__"))
    wrong_cost = _path_metrics()
    wrong_cost["__first_touch_round_trip_cost__"] = 0.02
    with pytest.raises(ValueError, match="single 1% round-trip cost"):
        build_path_primitives(wrong_cost)
    primitives = build_path_primitives(_path_metrics())
    invalid = SideTargetGeometry(0.60, 0.50, 8, 0.0, 0.01, 0.5, 0.2, 16.0, 0.2)
    with pytest.raises(ValueError, match="Unsupported exact first-passage geometry"):
        geometry_outcomes(primitives, invalid)


def test_exact_geometry_normalizes_unhit_nan_event_bars() -> None:
    frame = _path_metrics()
    frame.loc[2, "__bars_to_mfe_05r__"] = np.nan
    frame.loc[2, "__bars_to_mae_05r__"] = np.nan

    primitives = build_path_primitives(frame)

    assert primitives.bars_to_mfe_grid[2, 0] == -1.0
    assert primitives.bars_to_mae_grid[2, 0] == -1.0


def test_internal_folds_are_purged_and_never_use_apr_jun_oos() -> None:
    timestamps = pd.to_datetime(
        [
            "2025-04-01T00:00:00Z",
            "2025-12-01T00:00:00Z",
            "2025-12-31T22:00:00Z",
            "2026-01-01T00:00:00Z",
            "2026-02-01T00:00:00Z",
            "2026-03-01T00:00:00Z",
            "2026-03-31T23:00:00Z",
        ]
    )
    frame = pd.DataFrame({"__ts__": timestamps})
    folds = build_internal_chronological_folds(
        frame,
        validation_months=("2026-01", "2026-02", "2026-03"),
        purge_hours=25.0,
        min_train_rows=1,
        min_valid_rows=1,
    )

    assert len(folds) == 3
    for fold in folds:
        valid_ts = frame.iloc[fold["valid_idx"]]["__ts__"]
        train_ts = frame.iloc[fold["train_idx"]]["__ts__"]
        valid_start = valid_ts.min()
        assert valid_ts.max() < pd.Timestamp("2026-04-01T00:00:00Z")
        assert train_ts.max() < valid_start - pd.Timedelta(hours=25)


def _write_payload_cache(root, name: str, *, train_ts: list[str], valid_ts: list[str], full: bool) -> None:
    cache_dir = root / name
    cache_dir.mkdir(parents=True)
    train = pd.DataFrame({"__ts__": pd.to_datetime(train_ts, utc=True), "side": [1] * len(train_ts)})
    valid = pd.DataFrame({"__ts__": pd.to_datetime(valid_ts, utc=True), "side": [1] * len(valid_ts)})
    metrics_train = _path_metrics().iloc[:1].loc[np.zeros(len(train), dtype=int)].reset_index(drop=True)
    metrics_valid = _path_metrics().iloc[:1].loc[np.zeros(len(valid), dtype=int)].reset_index(drop=True)
    frames = {
        "train": train,
        "train_metrics": metrics_train,
        "valid": valid,
        "valid_metrics": metrics_valid,
        "x_train": pd.DataFrame({"f": np.arange(len(train), dtype=np.float32)}),
        "x_valid": pd.DataFrame({"f": np.arange(len(valid), dtype=np.float32)}),
    }
    paths = {}
    for key, frame in frames.items():
        path = cache_dir / f"{key}.parquet"
        frame.to_parquet(path, index=False)
        paths[key] = str(path)
    manifest = {
        "payload_paths": paths,
        "payload_train_sampling": "full_train_rows" if full else "bme_selection_sample",
        "valid_start": "2026-04-01T00:00:00+00:00" if full else "2026-06-26T00:00:00+00:00",
        "valid_end": "2026-07-01T00:00:00+00:00" if full else "2026-07-26T00:00:00+00:00",
    }
    (cache_dir / "fold_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_source_payload_separates_train_only_selection_from_main_fixed_oos(tmp_path) -> None:
    arm = tmp_path / "arm"
    _write_payload_cache(
        arm / "_feature_selection_phase" / "_fold_cache",
        "selection",
        train_ts=["2025-04-10", "2025-08-10", "2025-12-10", "2026-03-20", "2026-06-01"],
        valid_ts=["2026-06-26"],
        full=False,
    )
    _write_payload_cache(
        arm / "_fold_cache",
        "fixed_apr_jun",
        train_ts=["2025-04-10", "2025-08-10", "2025-12-10", "2026-03-20"],
        valid_ts=["2026-04-10", "2026-05-10", "2026-06-10"],
        full=True,
    )

    payloads = _load_source_payload(arm, purge_hours=25.0)
    selection_ts = pd.to_datetime(payloads["selection"]["train"]["__ts__"], utc=True)
    main_valid_ts = pd.to_datetime(payloads["main"]["valid"]["__ts__"], utc=True)

    assert selection_ts.max() < pd.Timestamp("2026-03-30T23:00:00Z")
    assert len(selection_ts) == 3  # equal begin/middle/end support, not June selection rows
    assert main_valid_ts.tolist() == list(pd.to_datetime(["2026-04-10", "2026-05-10", "2026-06-10"], utc=True))


def test_short_late_continuation_penalty_is_weak_edge_specific() -> None:
    primitives = build_path_primitives(_path_metrics())
    geometry = SideTargetGeometry(
        0.50,
        0.50,
        8,
        0.0,
        0.01,
        0.5,
        0.2,
        16.0,
        0.2,
        late_continuation_penalty=0.5,
    )
    pressure = np.ones(len(primitives.timeout_gross_return), dtype=np.float32)
    baseline = continuous_target(primitives, geometry)
    no_penalty = continuous_target(
        primitives,
        SideTargetGeometry(**{**geometry.__dict__, "late_continuation_penalty": 0.0}),
        late_continuation_pressure=pressure,
    )
    penalized = continuous_target(primitives, geometry, late_continuation_pressure=pressure)

    assert np.allclose(no_penalty, baseline)
    # Row zero is a stronger net path than the stopped row; the penalty must
    # preserve that ordering and hit the weak outcome more strongly.
    assert penalized[0] > penalized[1]
    assert penalized[1] / baseline[1] < penalized[0] / baseline[0]


def test_short_late_continuation_pressure_is_train_fitted_and_finite() -> None:
    n_rows = 96
    frame = pd.DataFrame(
        {
            "downside_deceleration_8h_rz": np.linspace(-2.0, 2.0, n_rows),
            "price_minus_oi_recovery_72h": np.linspace(-1.0, 3.0, n_rows),
            "climax_decay": np.linspace(0.0, 4.0, n_rows),
        }
    )
    frame.loc[0, "price_minus_oi_recovery_72h"] = np.nan
    pressure, features = short_late_continuation_pressure(frame, fit_indices=np.arange(80))

    assert features == (
        "downside_deceleration_8h_rz",
        "price_minus_oi_recovery_72h",
        "climax_decay",
    )
    assert pressure.dtype == np.float32
    assert np.isfinite(pressure).all()
    assert np.all((pressure >= 0.0) & (pressure <= 1.0))


def test_load_params_discards_numeric_hpo_objective(tmp_path) -> None:
    path = tmp_path / "hpo.json"
    path.write_text(
        json.dumps(
            {
                "objective": 1.5066,
                "loss_function": "regression",
                "num_leaves": 15,
                "mean_top10": 0.1,
            }
        ),
        encoding="utf-8",
    )

    assert _load_params(path) == {"objective": "regression_l2", "num_leaves": 15}
