import numpy as np
import pandas as pd

from extreme_price_movements.inference import live_meta_feature_overlays, run_inference
from extreme_price_movements.inference.live_meta_feature_overlays import (
    materialize_live_ae_gmm_features,
    materialize_live_source_regime_features,
)
from extreme_price_movements.inference.run_inference import (
    _fill_feature_from_refresh_preserve_existing,
)


def test_feature_refresh_only_fills_missing_values():
    frame = pd.DataFrame(
        {"feature": np.asarray([1.0, np.nan, 3.0], dtype=np.float32)},
        index=["a", "b", "c"],
    )
    refreshed = pd.Series([10.0, 20.0, 30.0], index=["a", "b", "c"])

    filled = _fill_feature_from_refresh_preserve_existing(
        frame,
        "feature",
        refreshed,
    )

    assert filled == 1
    np.testing.assert_allclose(frame["feature"], [1.0, 20.0, 3.0])


def test_feature_refresh_materializes_absent_column():
    frame = pd.DataFrame(index=["a", "b"])
    refreshed = pd.Series([4.0, 5.0], index=["a", "b"])

    filled = _fill_feature_from_refresh_preserve_existing(
        frame,
        "feature",
        refreshed,
    )

    assert filled == 2
    np.testing.assert_allclose(frame["feature"], [4.0, 5.0])


def test_source_regime_contract_includes_indirect_observable_inputs(monkeypatch):
    monkeypatch.setattr(
        run_inference,
        "_model_feature_contracts_for_audit",
        lambda *args, **kwargs: {
            "base_features": ["__regime_source_oi_agreement_score__"],
            "meta_features": [],
        },
    )

    contracts = run_inference._strategy_feature_contracts_from_orchestrator(
        object(),
        {"long_s52_meta_threshold_handoff": {"trade_side": "long"}},
    )

    selected = contracts["long_s52_meta_threshold_handoff"]
    assert "abs_ret_per_oi_z_24h" in selected
    assert "quote_volume_z_30d" in selected


def test_source_regime_batch_materialization_preserves_timestamp_and_symbol(
    monkeypatch,
):
    from scripts import materialize_candidate_source_tags as source_tags

    observed: dict[str, object] = {}

    monkeypatch.setattr(source_tags, "load_config", lambda _path: {})
    monkeypatch.setattr(
        source_tags,
        "build_feature_registry",
        lambda frame, config: {
            "available": {},
            "missing": {},
            "source_columns": ["signal"],
        },
    )
    monkeypatch.setattr(
        source_tags,
        "build_component_scores",
        lambda frame, registry, config: (pd.DataFrame(index=frame.index), {}),
    )

    def _archetypes(frame, components, registry, config):
        observed["timestamps"] = frame["__ts__"].copy()
        observed["symbols"] = frame["__symbol__"].copy()
        return pd.DataFrame(
            {
                "dirty_shock_avoid_score": np.zeros(len(frame), dtype=np.float32),
                "loud_breakout_impulse_score": np.linspace(
                    0.1, 0.4, len(frame), dtype=np.float32
                ),
            },
            index=frame.index,
        )

    monkeypatch.setattr(source_tags, "build_archetype_scores", _archetypes)
    timestamps = pd.to_datetime(
        ["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"], utc=True
    )
    frame = pd.DataFrame(
        {
            "__symbol__": ["A", "A"],
            "__ts__": timestamps,
            "signal": [1.0, 2.0],
        }
    )

    result = materialize_live_source_regime_features(
        frame,
        side="long",
        signal_bar_ts=None,
        required_columns=["__regime_source_breakout_impulse_score__"],
    )

    pd.testing.assert_series_equal(
        observed["timestamps"].reset_index(drop=True),
        pd.Series(timestamps, name="__ts__"),
    )
    assert observed["symbols"].tolist() == ["A", "A"]
    np.testing.assert_allclose(
        result["__regime_source_breakout_impulse_score__"], [0.1, 0.4]
    )


def test_ae_gmm_materialization_preserves_complete_frozen_outputs():
    frame = pd.DataFrame(
        {"dae_b16_00": np.asarray([0.25, -0.75], dtype=np.float32)},
        index=["a", "b"],
    )

    result = materialize_live_ae_gmm_features(
        frame,
        side="long",
        signal_bar_ts="2026-07-15T15:00:00Z",
        required_columns=["dae_b16_00"],
        state_payload={},
    )

    np.testing.assert_allclose(result["dae_b16_00"], [0.25, -0.75])


def test_ae_gmm_materialization_can_overwrite_store_outputs(monkeypatch):
    frame = pd.DataFrame(
        {"raw_state": [2.0, 3.0], "dae_b16_00": [99.0, 99.0]},
        index=["a", "b"],
    )
    payload = {
        "state": {"enabled": True, "feature_columns": ["raw_state"]},
        "state_path": "frozen.pkl",
    }
    monkeypatch.setattr(
        live_meta_feature_overlays,
        "transform_ae_gmm_features",
        lambda frame, state, index: pd.DataFrame(
            {"dae_b16_00": [0.25, 0.75]}, index=index
        ),
    )
    monkeypatch.setattr(
        live_meta_feature_overlays,
        "materialize_live_source_regime_features",
        lambda frame, **kwargs: frame,
    )

    result = materialize_live_ae_gmm_features(
        frame,
        side="long",
        signal_bar_ts="2026-07-15T15:00:00Z",
        required_columns=["dae_b16_00"],
        state_payload=payload,
        overwrite_existing=True,
    )

    np.testing.assert_allclose(result["dae_b16_00"], [0.25, 0.75])


def test_ae_gmm_materialization_rejects_incomplete_raw_rows(monkeypatch):
    frame = pd.DataFrame(
        {"raw_state": [2.0, np.nan], "dae_b16_00": [99.0, 99.0]},
        index=["complete", "incomplete"],
    )
    payload = {
        "state": {"enabled": True, "feature_columns": ["raw_state"]},
        "state_path": "frozen.pkl",
    }
    monkeypatch.setattr(
        live_meta_feature_overlays,
        "transform_ae_gmm_features",
        lambda frame, state, index: pd.DataFrame(
            {"dae_b16_00": np.full(len(index), 0.25)}, index=index
        ),
    )
    monkeypatch.setattr(
        live_meta_feature_overlays,
        "materialize_live_source_regime_features",
        lambda frame, **kwargs: frame,
    )

    result = materialize_live_ae_gmm_features(
        frame,
        side="long",
        signal_bar_ts="2026-07-15T15:00:00Z",
        required_columns=["dae_b16_00"],
        state_payload=payload,
        overwrite_existing=True,
    )

    assert result.at["complete", "dae_b16_00"] == 0.25
    assert np.isnan(result.at["incomplete", "dae_b16_00"])
