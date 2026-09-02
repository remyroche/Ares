from extreme_price_movements.inference.p8u_canonical_feature_adapter import (
    _selected_graph_dependency_closure,
    _restore_all_missing_regime_composites,
    canonical_features_from_saved_panel,
)
from extreme_price_movements.features import _ensure_frozen_market_spectral_parent_frames


def test_adapter_rejects_empty_universe():
    try:
        canonical_features_from_saved_panel({}, universe_symbols=(), requested_features=("x",))
    except ValueError as error:
        assert "non-empty full universe" in str(error)
    else:
        raise AssertionError("empty canonical universe must fail closed")


def test_adapter_rejects_missing_raw_symbol_before_feature_execution():
    import pandas as pd

    panel = {"close": pd.DataFrame({"A": [1.0]})}
    try:
        canonical_features_from_saved_panel(panel, universe_symbols=("A", "B"), requested_features=("x",))
    except ValueError as error:
        assert "misses canonical-universe symbols" in str(error)
    else:
        raise AssertionError("missing raw universe member must fail closed")


def test_adapter_defaults_to_the_full_canonical_feature_universe():
    import inspect

    signature = inspect.signature(canonical_features_from_saved_panel)
    assert signature.parameters["full_config_causal_universe"].default is True


def test_adapter_requires_a_scope_for_stateful_execution(tmp_path):
    import pandas as pd

    panel = {"close": pd.DataFrame({"A": [1.0]})}
    try:
        canonical_features_from_saved_panel(
            panel,
            universe_symbols=("A",),
            requested_features=("x",),
            state_dir=tmp_path,
        )
    except ValueError as error:
        assert "state_scope" in str(error)
    else:
        raise AssertionError("stateful adapter must require a semantic scope")


def test_selected_graph_closure_retains_frozen_spectral_source_parents():
    closure = _selected_graph_dependency_closure(
        ("model_field",),
        cfg={"MARKET_SPECTRAL_POSITION_SOURCE_FEATURE_KEYS": ("funding_z", "rv_24h")},
        frozen_dependencies=("helper_parent",),
    )
    assert closure == [
        "model_field",
        "rv_24h",
        "rv_120h",
        "helper_parent",
        "funding_z",
    ]


def test_selected_graph_closure_retains_regime_composite_parent():
    closure = _selected_graph_dependency_closure(
        ("q_tail_width__bars_in_high_vol_state_log_norm",),
        cfg={},
        frozen_dependencies=(),
    )
    assert closure == [
        "q_tail_width__bars_in_high_vol_state_log_norm",
        "bars_in_high_vol_state_log_norm",
        "rv_24h",
        "rv_120h",
    ]


def test_all_missing_regime_tail_composite_recovers_canonical_zero_fallback():
    import numpy as np
    import pandas as pd

    index = pd.DatetimeIndex(["2026-08-30T10:00:00Z"])
    columns = pd.Index(["A", "B"])
    generated = {
        "q_tail_width__bars_in_high_vol_state_log_norm": pd.DataFrame(
            np.nan, index=index, columns=columns
        )
    }
    _restore_all_missing_regime_composites(
        generated,
        requested=("q_tail_width__bars_in_high_vol_state_log_norm",),
        cfg={},
        index=index,
        columns=columns,
    )
    assert generated["q_tail_width__bars_in_high_vol_state_log_norm"].eq(0.0).all().all()


def test_frozen_spectral_parent_missing_frame_is_explicit_not_imputed(tmp_path):
    import json
    import numpy as np
    import pandas as pd

    state_path = tmp_path / "market_spectral_state.npz"
    state_path.write_text(
        json.dumps(
            {
                "schema": "strict_r3_market_spectral_source_state_v1",
                "source_keys": ["funding_per_hour", "funding_z"],
                "max_source_features": 64,
                "selected_columns": [
                    "funding_per_hour__mean",
                    "funding_z__std",
                ],
            }
        )
    )
    index = pd.DatetimeIndex(["2026-08-30T10:00:00Z"])
    columns = pd.Index(["A", "B"])
    existing = pd.DataFrame([[1.0, 2.0]], index=index, columns=columns)
    output = _ensure_frozen_market_spectral_parent_frames(
        {"funding_per_hour": existing},
        {
            "MARKET_SPECTRAL_POSITION_SOURCE_FEATURE_KEYS": [
                "funding_per_hour",
                "funding_z",
            ],
            "market_spectral_position_max_source_features": 64,
            "live_market_spectral_state_path": str(state_path),
        },
        feature_index=index,
        feature_columns=columns,
    )
    assert output["funding_per_hour"].equals(existing)
    assert output["funding_z"].shape == (1, 2)
    assert np.isnan(output["funding_z"].to_numpy()).all()
