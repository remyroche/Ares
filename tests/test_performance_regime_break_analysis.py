from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_performance_regime_break_analysis.py"
SPEC = importlib.util.spec_from_file_location("run_performance_regime_break_analysis", SCRIPT)
mod = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)


def test_compact_breakout_structure_feature_groups_are_capped() -> None:
    assert len(mod.BREAKOUT_STRUCTURE_FEATURE_GROUPS) == 7
    for features in mod.BREAKOUT_STRUCTURE_FEATURE_GROUPS.values():
        assert 1 <= len(features) <= 2
    assert len(mod.BREAKOUT_STRUCTURE_FEATURE_COLUMNS) <= 14
    assert "lower_lows_count_48h" in mod.BREAKOUT_STRUCTURE_FEATURE_COLUMNS


def test_spearman_redundancy_filter_keeps_one_representative_per_head(tmp_path: Path) -> None:
    x = np.linspace(-2.0, 2.0, 120, dtype=np.float32)
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-06-01", periods=len(x), freq="h", tz="UTC"),
            "symbol": ["AAA"] * len(x),
            "strong": x,
            "duplicate": x * 2.0 + 1.0,
            "independent": np.sin(np.arange(len(x), dtype=np.float32)),
        }
    )
    quality = pd.DataFrame(
        {
            "feature": ["strong", "duplicate", "independent"],
            "selection_score": [10.0, 1.0, 5.0],
        }
    )
    config = mod.AnalysisConfig(
        data_root=tmp_path,
        artifact_run_id="demo",
        output_dir=tmp_path,
        redundancy_abs_spearman_threshold=0.96,
        redundancy_max_rows=1_000,
    )

    kept, updated_quality, redundancy = mod._spearman_redundancy_filter(
        frame,
        ["strong", "duplicate", "independent"],
        quality,
        layer="base",
        strategy="demo",
        config=config,
    )

    assert "strong" in kept
    assert "duplicate" not in kept
    assert "independent" in kept
    dropped = redundancy.loc[redundancy["feature"].eq("duplicate")].iloc[0]
    assert bool(dropped["dropped_for_redundancy"])
    assert dropped["representative_feature"] == "strong"
    assert updated_quality.loc[updated_quality["feature"].eq("duplicate"), "redundancy_kept"].iloc[0] == False


def test_feature_store_hydration_cache_reuses_row_and_column_hash(tmp_path: Path) -> None:
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    ts = pd.date_range("2026-06-01", periods=4, freq="h", tz="UTC")
    pd.DataFrame({"f": np.arange(4, dtype=np.float32)}, index=ts).to_parquet(
        feature_dir / "symbol=AAA.parquet"
    )
    keys = pd.DataFrame({"timestamp": ts, "symbol": ["AAA"] * len(ts)})
    cache_dir = tmp_path / "cache"

    first = mod._hydrate_feature_store_for_keys(
        feature_dir,
        keys,
        ["f"],
        generate_url_composites=False,
        cache_dir=cache_dir,
        cache_enabled=True,
    )
    pd.DataFrame({"f": np.full(4, 99.0, dtype=np.float32)}, index=ts).to_parquet(
        feature_dir / "symbol=AAA.parquet"
    )
    second = mod._hydrate_feature_store_for_keys(
        feature_dir,
        keys,
        ["f"],
        generate_url_composites=False,
        cache_dir=cache_dir,
        cache_enabled=True,
    )

    assert first is not None and second is not None
    assert first["f"].tolist() == second["f"].tolist()
    assert len(list(cache_dir.glob("hydrated_feature_store_*.parquet"))) == 1


def test_timestamp_design_aggregation_collapses_symbol_rows() -> None:
    timestamps = pd.Series(
        pd.to_datetime(
            ["2026-06-01 00:00Z", "2026-06-01 00:00Z", "2026-06-01 01:00Z"],
            utc=True,
        )
    )
    X = np.asarray([[1.0, 3.0], [3.0, 5.0], [10.0, 20.0]], dtype=np.float32)
    groups = np.asarray([-1, -1, 0], dtype=np.int32)
    weights = np.asarray([1.0, 3.0, 2.0], dtype=np.float32)
    folds = np.asarray([0, 0, -1], dtype=np.int32)

    X_out, groups_out, weights_out, folds_out = mod._aggregate_design_by_timestamp(
        X,
        timestamps,
        groups,
        weights,
        folds,
    )

    assert X_out.shape == (2, 2)
    assert groups_out.tolist() == [-1, 0]
    assert np.allclose(X_out[0], [2.0, 4.0])
    assert np.isclose(weights_out[0], 2.0)
    assert folds_out.tolist() == [0, -1]


def test_bad_window_detection_uses_rows_per_day_threshold(tmp_path: Path) -> None:
    rolling = pd.DataFrame(
        {
            "start_day": pd.to_datetime(["2026-06-01", "2026-06-02"], utc=True),
            "end_day": pd.to_datetime(["2026-06-03", "2026-06-04"], utc=True),
            "n": [29.0, 30.0],
            "hit_rate_surprise_z": [-3.0, -3.0],
            "hit_rate_delta": [-0.20, -0.20],
        }
    )
    config = mod.AnalysisConfig(
        data_root=tmp_path,
        artifact_run_id="demo",
        output_dir=tmp_path,
        window_days=3,
        min_window_rows=0,
        min_window_rows_per_day=10.0,
        surprise_z_threshold=-1.5,
        hit_rate_delta_threshold=-0.0175,
    )

    bad = mod._detect_bad_windows(rolling, config=config)

    assert bad["n"].tolist() == [30.0]
    assert mod._effective_min_window_rows(config) == 30.0


def test_breakout_head_summary_includes_bad_day_share_and_severity(tmp_path: Path) -> None:
    support = pd.DataFrame(
        {
            "layer": ["meta"],
            "strategy": ["demo"],
            "slice": ["top30"],
            "support_days": [10],
            "eligible_support_days": [3],
            "support_rows": [200],
            "eligible_support_day_list": ["2026-06-01|2026-06-03|2026-06-10"],
        }
    )
    bad_windows = pd.DataFrame(
        {
            "layer": ["meta", "meta"],
            "strategy": ["demo", "demo"],
            "slice": ["top30", "top30"],
            "start_day": pd.to_datetime(["2026-06-01", "2026-06-02"], utc=True),
            "end_day": pd.to_datetime(["2026-06-03", "2026-06-04"], utc=True),
            "hit_rate_delta": [-0.05, -0.10],
            "hit_rate_surprise_z": [-2.0, -4.0],
            "window_severity": [1.0, 3.0],
        }
    )
    episodes = pd.DataFrame(
        {
            "layer": ["meta"],
            "strategy": ["demo"],
            "slice": ["top30"],
            "start_day": pd.to_datetime(["2026-06-01"], utc=True),
            "end_day": pd.to_datetime(["2026-06-04"], utc=True),
            "episode_hit_rate_delta": [-0.08],
            "episode_hit_rate_surprise_z": [-5.0],
            "episode_breakout_weight": [4.0],
        }
    )

    out = mod._aggregate_breakout_head_summary(support, bad_windows, episodes)
    row = out.iloc[0]

    assert row["bad_day_count"] == 2
    assert np.isclose(row["bad_day_share"], 2.0 / 3.0)
    assert row["window_severity_sum"] == 4.0
    assert np.isclose(row["window_severity_weighted_hr_delta"], -0.0875)
    assert row["episode_breakout_weight_sum"] == 4.0


def test_bad_window_threshold_calibration_targets_support_day_share(tmp_path: Path) -> None:
    days = pd.date_range("2026-01-01", periods=20, freq="D", tz="UTC")
    rolling = pd.DataFrame(
        {
            "start_day": days,
            "end_day": days,
            "window_end_day": days,
            "window_days": np.ones(len(days)),
            "n": np.full(len(days), 50.0),
            "hit_rate_surprise_z": -np.linspace(0.1, 5.0, len(days)),
            "hit_rate_delta": -np.linspace(0.001, 0.10, len(days)),
            "surprise_z_threshold": np.full(len(days), -0.1),
            "hit_rate_delta_threshold": np.full(len(days), -0.001),
        }
    )
    config = mod.AnalysisConfig(
        data_root=tmp_path,
        artifact_run_id="demo",
        output_dir=tmp_path,
        window_days=1,
        min_window_rows=1,
        min_window_rows_per_day=1.0,
        surprise_z_threshold=-10.0,
        hit_rate_delta_threshold=-0.50,
        bad_window_calibration_enabled=True,
        target_bad_day_share=0.15,
        bad_window_calibration_grid_size=20,
    )

    diagnostics, mask = mod._calibrate_bad_window_thresholds(
        rolling,
        eligible_days={pd.Timestamp(day).floor("D") for day in days},
        config=config,
    )

    assert diagnostics["calibration_status"] == "ok"
    assert diagnostics["calibration_realized_bad_day_share"] <= 0.15
    assert mask.sum() > 0


def test_rolling_state_transforms_emit_slope_accel_and_exposure() -> None:
    ts = pd.date_range("2026-06-01", periods=36, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": list(ts) * 2,
            "symbol": ["AAA"] * len(ts) + ["BBB"] * len(ts),
            "state": np.r_[np.linspace(0, 3, len(ts)), np.linspace(1, -2, len(ts))],
        }
    )

    out = mod._generate_rolling_state_transform_features(
        frame,
        ["state"],
        windows=(4,),
        extreme_z=1.0,
    )

    assert "roll_slope_w4__state" in out.columns
    assert "roll_accel_w4__state" in out.columns
    assert "extreme_exposure_w4__state" in out.columns
    assert np.isfinite(out["roll_slope_w4__state"]).sum() > 0
    assert np.nanmax(out["extreme_exposure_w4__state"].to_numpy()) <= 1.0


def test_advanced_covariance_diagnostics_have_expected_keys() -> None:
    rng = np.random.default_rng(7)
    base = rng.normal(size=(180, 8)).astype(np.float32)
    bad = rng.normal(size=(60, 8)).astype(np.float32)
    bad[:, 1] = bad[:, 0] * 0.8 + rng.normal(scale=0.2, size=60)

    cov_stats = mod._matrix_break_reference_stats(base, bad, kind="cov")
    precision = mod._precision_shift_diagnostics(base, bad, max_features=8)
    tail = mod._tail_coexceedance_diagnostics(base, bad)
    nonlinear = mod._nonlinear_dependence_diagnostics(base, bad, max_features=6)

    assert "historical_cov_break_z" in cov_stats
    assert precision["precision_status"] in {
        "insufficient_observations",
        "ridge_precision",
    } or str(precision["precision_status"]).startswith("failed:")
    assert "tail_coexceedance_frobenius_shift" in tail
    assert "distance_corr_frobenius_shift" in nonlinear


def test_ebm_interaction_diagnostics_runs_leave_one_episode_out(tmp_path: Path) -> None:
    pytest.importorskip("interpret")
    n = 480
    ts = pd.date_range("2026-06-01", periods=n, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": ts,
            "symbol": ["AAA"] * n,
            "f1": np.random.default_rng(1).normal(size=n),
            "f2": np.random.default_rng(2).normal(size=n),
            "oof_prob": np.full(n, 0.55),
            "y_bin": np.zeros(n),
        }
    )
    episode_specs = [
        (0, pd.Timestamp("2026-06-05", tz="UTC"), pd.Timestamp("2026-06-07", tz="UTC"), -4.0),
        (1, pd.Timestamp("2026-06-12", tz="UTC"), pd.Timestamp("2026-06-14", tz="UTC"), -3.5),
    ]
    episodes = []
    for eid, start, end, z in episode_specs:
        mask = frame["timestamp"].dt.floor("D").between(start, end)
        frame.loc[mask, "f1"] += 3.0
        frame.loc[mask, "f2"] += 3.0
        frame.loc[mask, "y_bin"] = 1.0
        episodes.append(
            {
                "episode_id": eid,
                "start_day": start,
                "end_day": end,
                "min_hit_rate_surprise_z": z,
                "min_hit_rate_delta": -0.05,
                "window_count": 2,
                "total_rows_in_bad_windows": int(mask.sum()),
            }
        )
    feature_scores = pd.DataFrame(
        [
            {
                "breakout_key": f"base|demo|all|{eid}",
                "feature": feature,
                "candidate_score": 0.1,
                "episode_explanation_score": 1.0,
                "regime_candidate": True,
            }
            for eid, *_ in episode_specs
            for feature in ("f1", "f2")
        ]
    )
    config = mod.AnalysisConfig(
        data_root=tmp_path,
        artifact_run_id="demo",
        output_dir=tmp_path,
        ebm_max_episodes=2,
        ebm_max_features=2,
        ebm_max_pairs=1,
        ebm_max_rows_per_episode=80,
        ebm_max_control_rows=160,
        ebm_max_rounds=20,
        ebm_min_rows=80,
    )

    pairs, thresholds = mod._episode_ebm_interaction_diagnostics(
        frame,
        strategy="demo",
        layer="base",
        slice_name="all",
        episodes=pd.DataFrame(episodes),
        feature_scores=feature_scores,
        slice_mask=np.ones(len(frame), dtype=bool),
        config=config,
    )

    assert not pairs.empty
    assert not thresholds.empty
    assert {"loeo_selection_frequency", "loeo_delta_logloss_mean", "false_alarm_rate_control_mean"}.issubset(
        pairs.columns
    )


def test_streamed_transform_cache_saves_and_reuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-06-01", periods=12, freq="h", tz="UTC"),
            "symbol": ["AAA"] * 12,
            "raw_state": np.linspace(-1.0, 1.0, 12, dtype=np.float32),
        }
    )
    raw_screen = pd.DataFrame(
        {
            "feature": ["raw_state"],
            "feature_family": ["demo"],
            "raw_breakout_link_score": [0.5],
            "selected_for_operator_generation": [True],
        }
    )
    config = mod.AnalysisConfig(
        data_root=tmp_path,
        artifact_run_id="demo_run",
        output_dir=tmp_path / "out",
        transform_cache_dir=tmp_path / "transform_cache",
    )
    calls = {"count": 0}

    def fake_expected(*args, **kwargs):
        return ["cached_transform__raw_state"]

    def fake_generate(panel, raw_screen, *, config):
        calls["count"] += 1
        return pd.DataFrame(
            {"cached_transform__raw_state": np.arange(len(panel), dtype=np.float32)},
            index=panel.index,
        )

    monkeypatch.setattr(mod, "_expected_breakout_generated_columns", fake_expected)
    monkeypatch.setattr(mod, "_generate_breakout_exploration_composites", fake_generate)

    first = mod._append_streamed_generated_features(
        frame,
        config=config,
        feature_columns=["raw_state"],
        raw_breakout_screen=raw_screen,
    )
    second = mod._append_streamed_generated_features(
        frame,
        config=config,
        feature_columns=["raw_state"],
        raw_breakout_screen=raw_screen,
    )

    assert calls["count"] == 1
    assert "cached_transform__raw_state" in first.columns
    assert "cached_transform__raw_state" in second.columns
    assert first["cached_transform__raw_state"].equals(second["cached_transform__raw_state"])
    assert len(list((tmp_path / "transform_cache").glob("generated_transforms_*.parquet"))) == 1


def test_previous_meta_parent_loader_keeps_only_meta_raw_features(tmp_path: Path) -> None:
    report = tmp_path / "feature_breakout_explanatory_strength_by_head.csv"
    pd.DataFrame(
        {
            "layer": ["meta", "meta", "meta", "base", "meta"],
            "strategy": ["s1", "s1", "s1", "s1", "s2"],
            "slice": ["top30", "top30", "top30", "top30", "all"],
            "feature": [
                "mkt_oi_z_30d",
                "xs_mean__mkt_oi_z_30d",
                "oof_uncertainty_score",
                "amihud_z",
                "vol_z_4h",
            ],
            "breakout_explanatory_strength": [0.9, 1.0, 0.8, 2.0, 0.7],
            "explained_breakout_weight_share": [0.5, 0.5, 0.5, 0.5, 0.5],
            "explained_breakout_count": [3, 3, 3, 3, 3],
        }
    ).to_csv(report, index=False)
    config = mod.AnalysisConfig(
        data_root=tmp_path,
        artifact_run_id="demo",
        output_dir=tmp_path,
        previous_meta_parent_report=report,
        previous_meta_parent_top_n=50,
        previous_meta_parent_slice="top30",
    )

    parent_map, selected = mod._load_previous_meta_parent_features(report, config=config)

    assert parent_map == {("meta", "s1", "top30"): ["mkt_oi_z_30d"]}
    assert selected["feature"].tolist() == ["mkt_oi_z_30d"]
    assert mod._previous_meta_parent_features_for_head(
        parent_map,
        layer="base",
        strategy="s1",
        config=config,
    ) == []
    assert mod._previous_meta_parent_features_for_head(
        parent_map,
        layer="meta",
        strategy="s1",
        config=config,
    ) == ["mkt_oi_z_30d"]


def test_previous_meta_parent_screen_blocks_generated_feature_recursion() -> None:
    raw_screen = pd.DataFrame(
        {
            "feature": ["raw_screen_feature"],
            "feature_family": ["other"],
            "raw_breakout_link_score": [0.1],
            "selected_for_operator_generation": [True],
        }
    )

    merged = mod._merge_previous_meta_parent_screen(
        raw_screen,
        [
            "mkt_oi_z_30d",
            "xs_mean__mkt_oi_z_30d",
            "corr_w24__a__b",
            "oof_prob",
        ],
    )

    assert "mkt_oi_z_30d" in merged["feature"].tolist()
    assert "raw_screen_feature" in merged["feature"].tolist()
    assert "xs_mean__mkt_oi_z_30d" not in merged["feature"].tolist()
    assert "corr_w24__a__b" not in merged["feature"].tolist()
    assert "oof_prob" not in merged["feature"].tolist()
    row = merged.loc[merged["feature"].eq("mkt_oi_z_30d")].iloc[0]
    assert bool(row["selected_for_operator_generation"])
    assert row["operator_generation_source"] == "previous_meta_parent"


def test_single_generated_transform_cache_appends_missing_columns(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-06-01", periods=6, freq="h", tz="UTC"),
            "symbol": ["AAA"] * 6,
            "raw_a": np.linspace(0.0, 1.0, 6, dtype=np.float32),
            "raw_b": np.linspace(1.0, 2.0, 6, dtype=np.float32),
        }
    )
    config = mod.AnalysisConfig(
        data_root=tmp_path,
        artifact_run_id="demo_run",
        output_dir=tmp_path / "out",
        transform_cache_dir=tmp_path / "transform_cache",
        generated_transform_cache_ttl_days=0.0,
        generated_transform_cache_keep_last_n=1,
    )

    path_a, meta_a = mod._transform_cache_paths(
        frame,
        config=config,
        feature_columns=["raw_a"],
        raw_features=["raw_a"],
        mode="breakout",
    )
    path_b, meta_b = mod._transform_cache_paths(
        frame,
        config=config,
        feature_columns=["raw_a", "raw_b"],
        raw_features=["raw_a", "raw_b"],
        mode="breakout",
    )

    assert path_a == path_b
    assert path_a is not None
    mod._write_transform_cache(
        path_a,
        frame,
        pd.DataFrame({"generated_a": np.arange(len(frame), dtype=np.float32)}, index=frame.index),
        meta_a,
    )
    mod._write_transform_cache(
        path_b,
        frame,
        pd.DataFrame({"generated_b": np.arange(len(frame), dtype=np.float32) + 10.0}, index=frame.index),
        meta_b,
    )

    cached = pd.read_parquet(path_a)
    assert {"generated_a", "generated_b"}.issubset(cached.columns)
    assert len(list((tmp_path / "transform_cache").glob("generated_transforms_*.parquet"))) == 1


def test_streamed_transform_cache_regenerates_when_row_coverage_is_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame_a = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-06-01", periods=4, freq="h", tz="UTC"),
            "symbol": ["AAA"] * 4,
            "raw_state": np.linspace(0.0, 1.0, 4, dtype=np.float32),
        }
    )
    frame_b = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2026-06-01 00:00", tz="UTC"),
                pd.Timestamp("2026-06-01 01:00", tz="UTC"),
                pd.Timestamp("2026-06-02 00:00", tz="UTC"),
                pd.Timestamp("2026-06-02 01:00", tz="UTC"),
            ],
            "symbol": ["AAA", "AAA", "BBB", "BBB"],
            "raw_state": np.linspace(1.0, 2.0, 4, dtype=np.float32),
        }
    )
    raw_screen = pd.DataFrame(
        {
            "feature": ["raw_state"],
            "feature_family": ["demo"],
            "raw_breakout_link_score": [0.5],
            "selected_for_operator_generation": [True],
        }
    )
    config = mod.AnalysisConfig(
        data_root=tmp_path,
        artifact_run_id="demo_run",
        output_dir=tmp_path / "out",
        transform_cache_dir=tmp_path / "transform_cache",
    )
    calls = {"count": 0}

    def fake_expected(*args, **kwargs):
        return ["cached_transform__raw_state"]

    def fake_generate(panel, raw_screen, *, config):
        calls["count"] += 1
        return pd.DataFrame(
            {"cached_transform__raw_state": np.arange(len(panel), dtype=np.float32) + calls["count"] * 10.0},
            index=panel.index,
        )

    monkeypatch.setattr(mod, "_expected_breakout_generated_columns", fake_expected)
    monkeypatch.setattr(mod, "_generate_breakout_exploration_composites", fake_generate)

    first = mod._append_streamed_generated_features(
        frame_a,
        config=config,
        feature_columns=["raw_state"],
        raw_breakout_screen=raw_screen,
    )
    second = mod._append_streamed_generated_features(
        frame_b,
        config=config,
        feature_columns=["raw_state"],
        raw_breakout_screen=raw_screen,
    )

    assert calls["count"] == 2
    assert "cached_transform__raw_state" in first.columns
    assert "cached_transform__raw_state" in second.columns
    cache_files = list((tmp_path / "transform_cache").glob("generated_transforms_*.parquet"))
    assert len(cache_files) == 1
    cached = pd.read_parquet(cache_files[0])
    assert "__cache_scope" in cached.columns
    assert cached["__cache_scope"].nunique() == 2
    assert len(cached) == len(frame_a) + len(frame_b)


def test_generated_transform_cache_append_cap_skips_large_append(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-06-01", periods=4, freq="h", tz="UTC"),
            "symbol": ["AAA"] * 4,
            "raw_state": np.linspace(0.0, 1.0, 4, dtype=np.float32),
        }
    )
    frame_next = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-06-02", periods=4, freq="h", tz="UTC"),
            "symbol": ["BBB"] * 4,
            "raw_state": np.linspace(1.0, 2.0, 4, dtype=np.float32),
        }
    )
    config = mod.AnalysisConfig(
        data_root=tmp_path,
        artifact_run_id="demo_run",
        output_dir=tmp_path / "out",
        transform_cache_dir=tmp_path / "transform_cache",
        generated_transform_cache_max_rows=6,
    )
    path, meta = mod._transform_cache_paths(
        frame,
        config=config,
        feature_columns=["raw_state"],
        raw_features=["raw_state"],
        mode="breakout",
    )
    assert path is not None
    assert mod._write_transform_cache(
        path,
        frame,
        pd.DataFrame({"generated_a": np.arange(len(frame), dtype=np.float32)}, index=frame.index),
        meta,
    )
    _, meta_next = mod._transform_cache_paths(
        frame_next,
        config=config,
        feature_columns=["raw_state"],
        raw_features=["raw_state"],
        mode="breakout",
    )
    assert not mod._write_transform_cache(
        path,
        frame_next,
        pd.DataFrame({"generated_a": np.arange(len(frame_next), dtype=np.float32)}, index=frame_next.index),
        meta_next,
    )
    cached = pd.read_parquet(path)
    assert len(cached) == len(frame)
