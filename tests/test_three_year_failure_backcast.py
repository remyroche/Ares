import json
from argparse import Namespace

import numpy as np
import pandas as pd

from scripts.backfill_complete_july_meta_predictions import (
    _hourly_close_proxy_outcomes,
)
from scripts.materialize_three_year_failure_backcast import (
    _month_chunks,
    _valid_completed_chunk,
)
from scripts.run_failure_episode_taxonomy import (
    _failure_mode_composition_audit,
    _load_source,
    _local_model_error_shape,
    _mixture_profiles,
    _negative_day_mode_catalog,
    parse_args,
)


def _source_rows(day: str) -> pd.DataFrame:
    timestamp = pd.Timestamp(day, tz="UTC")
    return pd.DataFrame(
        {
            "__ts__": [timestamp],
            "__symbol__": ["BTC/USD:USD"],
            "side_name": ["long"],
            "archetype_policy_key": ["long_test"],
            "hit_probability": [0.8],
            "clean_exec": [1.0],
            "ev_after_1pct": [0.01],
            "selected_for_monitor": [True],
            "selected_top30": [True],
        }
    )


def test_month_chunks_preserve_partial_boundaries() -> None:
    start = pd.Timestamp("2023-07-18T00:00:00Z")
    end = pd.Timestamp("2023-09-03T00:00:00Z")

    chunks = _month_chunks(start, end)

    assert [(chunk.start, chunk.end) for chunk in chunks] == [
        (start, pd.Timestamp("2023-08-01T00:00:00Z")),
        (
            pd.Timestamp("2023-08-01T00:00:00Z"),
            pd.Timestamp("2023-09-01T00:00:00Z"),
        ),
        (pd.Timestamp("2023-09-01T00:00:00Z"), end),
    ]


def test_failure_taxonomy_loads_monthly_source_directory(tmp_path) -> None:
    _source_rows("2024-01-01").to_parquet(
        tmp_path / "candidates_202401.parquet", index=False
    )
    _source_rows("2024-02-01").to_parquet(
        tmp_path / "candidates_202402.parquet", index=False
    )
    args = Namespace(
        ledger=tmp_path,
        start="",
        end="",
        include_all_rows=False,
        required_years=3.0,
        provenance="frozen_backcast_diagnostic",
    )

    frame, manifest = _load_source(args)

    assert len(frame) == 2
    assert manifest["source_shards"] == 2
    assert manifest["parquet_rows"] == 2
    assert manifest["provenance"] == "frozen_backcast_diagnostic"


def test_taxonomy_cli_does_not_reuse_repository_root_by_default(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["run_failure_episode_taxonomy.py"])

    args = parse_args()

    assert args.reuse_local_taxonomy_root is None


def test_resume_rejects_pre_attribution_monthly_schema(tmp_path) -> None:
    chunk = _month_chunks(
        pd.Timestamp("2024-01-01T00:00:00Z"),
        pd.Timestamp("2024-02-01T00:00:00Z"),
    )[0]
    _source_rows("2024-01-01").to_parquet(
        tmp_path / "frozen_predictions.parquet", index=False
    )
    manifest = {
        "start": chunk.start.isoformat(),
        "end_exclusive": chunk.end.isoformat(),
        "rows": 1,
        "return_unit": "decimal_notional_return",
        "cost_counted_once": True,
        "outcome_contract_version": "hourly_close_policy_proxy_v2_activation_deadline",
        "policy_bar_minutes": 15,
        "path_stats": {"long": {"coverage": 1.0}},
        "observable_feature_names": ["gmm_entropy"],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    assert not _valid_completed_chunk(tmp_path, chunk, min_path_coverage=0.90)

    manifest["observable_feature_names"].append("base_attr_bias")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert _valid_completed_chunk(tmp_path, chunk, min_path_coverage=0.90)


def test_local_error_shape_separates_base_and_meta_residuals() -> None:
    source = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(["2025-01-01"] * 4, utc=True),
            "side_name": "long",
            "archetype_policy_key": "trend",
            "clean_exec": [1.0, 0.0, 1.0, 0.0],
            "base_score": [0.9, 0.8, 0.7, 0.6],
            "score_meta_base_soft_label": [0.8, 0.9, 0.6, 0.7],
            "historical_rank": [1.0, 0.75, 0.5, 0.25],
            "ev_after_1pct": [0.03, -0.02, 0.01, -0.01],
            "dirty_positive": [0.0, 0.0, 1.0, 0.0],
            "first_touch_bad_mae_1r": [0.0, 1.0, 0.0, 1.0],
            "full_path_bad_mae_1r": [0.0, 1.0, 1.0, 1.0],
        }
    )

    result = _local_model_error_shape(source)

    assert len(result) == 1
    assert result.loc[0, "expost__base_false_positive_rate"] == 0.5
    assert result.loc[0, "expost__meta_false_positive_rate"] == 0.5
    assert result.loc[0, "expost__base_meta_disagreement"] > 0.0
    assert result.loc[0, "expost__ranking_spearman"] > 0.0


def test_local_error_shape_preserves_missing_meta_evidence() -> None:
    source = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(["2025-01-01"] * 4, utc=True),
            "side_name": "long",
            "archetype_policy_key": "trend",
            "clean_exec": [1.0, 0.0, 1.0, 0.0],
            "base_score": [0.9, 0.8, 0.7, 0.6],
            "score_meta_base_soft_label": [np.nan] * 4,
            "historical_rank": [1.0, 0.75, 0.5, 0.25],
            "ev_after_1pct": [0.03, -0.02, 0.01, -0.01],
            "dirty_positive": [0.0, 0.0, 1.0, 0.0],
            "first_touch_bad_mae_1r": [0.0, 1.0, 0.0, 1.0],
            "full_path_bad_mae_1r": [0.0, 1.0, 1.0, 1.0],
        }
    )

    result = _local_model_error_shape(source)

    assert result.loc[0, "expost__base_false_positive_rate"] == 0.5
    assert pd.isna(result.loc[0, "expost__meta_false_positive_rate"])
    assert pd.isna(result.loc[0, "expost__meta_correction_rate"])
    assert pd.isna(result.loc[0, "expost__base_meta_sign_disagreement_rate"])
    assert pd.isna(result.loc[0, "expost__base_meta_disagreement"])


def test_negative_day_catalog_keeps_parent_and_local_failure_modes() -> None:
    day = pd.Timestamp("2025-02-03", tz="UTC")
    daily = pd.DataFrame(
        {
            "day": [day],
            "negative_pnl_day": [True],
            "net_ev": [-0.04],
            "mean_ev": [-0.01],
            "selected_rows": [4],
            "distinct_assets": [3],
        }
    )
    parent_calendar = pd.DataFrame(
        {
            "day": [day],
            "side_name": ["global"],
            "archetype_policy_key": ["global_market"],
            "event_block": ["event_001"],
        }
    )
    assignment_columns = {
        "method": ["pca_student_t"],
        "latent_dim": [2],
        "clusters": [2],
        "cluster_id": [1],
        "cluster_posterior_max": [0.9],
        "cluster_entropy": [0.2],
    }
    parent_assignment = pd.DataFrame(
        {
            "side_name": ["global"],
            "archetype_policy_key": ["global_market"],
            "event_block": ["event_001"],
            "event_start": [day],
            "event_end": [day],
            **assignment_columns,
        }
    )
    parent_profile = pd.DataFrame(
        {
            "side_name": ["global"],
            "archetype_policy_key": ["global_market"],
            "method": ["pca_student_t"],
            "latent_dim": [2],
            "clusters": [2],
            "cluster_id": [1],
            "semantic_label": ["ranking_collapse__liquidation_pressure"],
        }
    )
    local_calendar = pd.DataFrame(
        {
            "day": [day],
            "side_name": ["short"],
            "archetype_policy_key": ["short_breakout"],
            "event_block": ["event_004"],
            "adverse_event": [True],
        }
    )
    local_assignment = parent_assignment.assign(
        side_name="short",
        archetype_policy_key="short_breakout",
        event_block="event_004",
    )
    local_profile = parent_profile.assign(
        side_name="short",
        archetype_policy_key="short_breakout",
        semantic_label="overconfident_false_positive__funding_transition",
    )

    result = _negative_day_mode_catalog(
        daily,
        parent_calendar,
        parent_assignment,
        parent_profile,
        local_calendar,
        local_assignment,
        local_profile,
    )

    assert len(result) == 1
    assert bool(result.loc[0, "parent_mode_assigned"])
    assert bool(result.loc[0, "all_active_local_modes_assigned"])
    assert result.loc[0, "active_local_failure_cells"] == 1
    assert "short_breakout" in result.loc[0, "local_failure_modes"]


def test_failure_mode_composition_flags_symbol_concentration() -> None:
    days = pd.date_range("2025-02-01", periods=4, freq="D", tz="UTC")
    source = pd.DataFrame(
        {
            "__ts__": days,
            "__symbol__": ["BTC/USD:USD"] * 4,
            "side_name": ["short"] * 4,
            "archetype_policy_key": ["short_breakout"] * 4,
        }
    )
    calendar = pd.DataFrame(
        {
            "day": days,
            "side_name": ["short"] * 4,
            "archetype_policy_key": ["short_breakout"] * 4,
            "event_block": ["event_001"] * 4,
            "adverse_event": [True] * 4,
        }
    )
    assignments = pd.DataFrame(
        {
            "side_name": ["short"],
            "archetype_policy_key": ["short_breakout"],
            "event_block": ["event_001"],
            "method": ["pca_gmm"],
            "latent_dim": [2],
            "clusters": [2],
            "cluster_id": [0],
        }
    )
    profiles = pd.DataFrame(
        {
            "side_name": ["short"],
            "archetype_policy_key": ["short_breakout"],
            "method": ["pca_gmm"],
            "latent_dim": [2],
            "clusters": [2],
            "cluster_id": [0],
            "semantic_label": ["ranking_collapse__liquidation_pressure"],
        }
    )

    report = _failure_mode_composition_audit(
        source,
        calendar,
        assignments,
        profiles,
        parent_scope=False,
    )

    assert len(report) == 1
    assert report.loc[0, "distinct_symbols"] == 1
    assert report.loc[0, "dominant_symbol_fraction"] == 1.0
    assert bool(report.loc[0, "composition_redundancy_warning"])


def test_mixture_profile_separates_error_shape_from_market_mechanism() -> None:
    day = pd.Timestamp("2025-02-01", tz="UTC")
    taxonomy = pd.DataFrame(
        {
            "side_name": ["short"],
            "archetype_policy_key": ["short_breakout"],
            "event_block": ["event_001"],
            "event_start": [day],
            "event_end": [day],
            "calendar_mean_ev": [-0.02],
            "calendar_mean_signed_surprise": [-0.12],
            "calendar_error__ranking_spearman": [-0.20],
            "calendar_error__meta_false_positive_rate": [0.80],
            "family__error_vector__active__meta_false_positive_rate": [9.0],
            "family__liquidation_pressure__peak_abs_z": [2.0],
        }
    )
    assignments = pd.DataFrame(
        {
            "source_index": [0],
            "side_name": ["short"],
            "archetype_policy_key": ["short_breakout"],
            "event_block": ["event_001"],
            "event_start": [day],
            "event_end": [day],
            "method": ["pca_student_t"],
            "latent_dim": [2],
            "clusters": [2],
            "cluster_id": [1],
            "cluster_posterior_max": [0.9],
            "cluster_entropy": [0.1],
        }
    )

    profile = _mixture_profiles(taxonomy, assignments)

    assert profile.loc[0, "semantic_label"] == (
        "directional_inversion__liquidation_pressure"
    )


def test_hourly_proxy_respects_native_policy_activation_deadline(
    monkeypatch, tmp_path
) -> None:
    entry = pd.Timestamp("2025-01-01T00:00:00Z")
    index = pd.date_range(entry, periods=7, freq="h")
    closes = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0, 102.0, 101.5], index=index)

    def fake_load(self, symbol, *, columns, start_ts, end_ts):
        del self, symbol, columns, start_ts, end_ts
        return pd.DataFrame({"close": closes})

    monkeypatch.setattr(
        "extreme_price_movements.data_store.PartitionedOHLCVStore.load",
        fake_load,
    )
    rows = pd.DataFrame(
        {
            "__ts__": [entry],
            "__symbol__": ["BTC/USD:USD"],
            "side_name": ["long"],
            "archetype_policy_key": ["long_test"],
            "__barrier_pct__": [0.01],
        }
    )
    policy = {
        "default": {
            "long": {
                "policy_key": "long_test",
                "tp_r": 1.0,
                "sl_r": 1.0,
                "trail_r": 0.25,
                "max_bars_to_mfe": 16.0,
            }
        }
    }

    native_15m, stats = _hourly_close_proxy_outcomes(
        rows,
        feature_root=tmp_path / "features/run",
        policy_manifest=policy,
        horizon_hours=6,
        policy_bar_minutes=15,
    )
    hourly_policy, _ = _hourly_close_proxy_outcomes(
        rows,
        feature_root=tmp_path / "features/run",
        policy_manifest=policy,
        horizon_hours=6,
        policy_bar_minutes=60,
    )

    assert native_15m.loc[0, "timeout"] == 1.0
    assert native_15m.loc[0, "clean_exec"] == 0.0
    assert hourly_policy.loc[0, "timeout"] == 0.0
    assert hourly_policy.loc[0, "clean_exec"] == 1.0
    assert stats["outcome_contract_version"].endswith("activation_deadline")
    assert np.isfinite(native_15m.loc[0, "ev_after_1pct"])


def test_hourly_proxy_separates_unlisted_symbols_from_path_gaps(
    monkeypatch, tmp_path
) -> None:
    listed_entry = pd.Timestamp("2025-01-02T00:00:00Z")
    history = pd.Series(
        100.0,
        index=pd.date_range(listed_entry, periods=4, freq="h"),
    )

    def fake_load(self, symbol, *, columns, start_ts, end_ts):
        del self, symbol, columns, start_ts, end_ts
        return pd.DataFrame({"close": history})

    monkeypatch.setattr(
        "extreme_price_movements.data_store.PartitionedOHLCVStore.load",
        fake_load,
    )
    rows = pd.DataFrame(
        {
            "__ts__": [listed_entry - pd.Timedelta(days=1), listed_entry],
            "__symbol__": ["BTC/USD:USD", "BTC/USD:USD"],
            "side_name": ["long", "long"],
            "archetype_policy_key": ["long_test", "long_test"],
            "__barrier_pct__": [0.01, 0.01],
        }
    )
    policy = {
        "default": {
            "long": {
                "policy_key": "long_test",
                "tp_r": 1.0,
                "sl_r": 1.0,
                "trail_r": 0.25,
                "max_bars_to_mfe": 16.0,
            }
        }
    }

    outcomes, stats = _hourly_close_proxy_outcomes(
        rows,
        feature_root=tmp_path / "features/run",
        policy_manifest=policy,
        horizon_hours=2,
        policy_bar_minutes=15,
    )

    assert outcomes["ev_after_1pct"].notna().sum() == 1
    assert stats["raw_candidate_rows"] == 2
    assert stats["historically_tradable_rows"] == 1
    assert stats["unavailable_contract_rows"] == 1
    assert stats["internal_gap_rows"] == 0
    assert stats["coverage"] == 1.0
    assert stats["raw_universe_coverage"] == 0.5
