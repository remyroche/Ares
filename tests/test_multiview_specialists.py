from __future__ import annotations

import numpy as np
import pandas as pd

from extreme_price_movements.multiview_specialists import (
    apply_synergy_features, discover_opportunity_views, is_permitted_feature,
    opportunity_conditioned_synergy, permitted_causal_features,
)


def test_data_discovery_uses_opportunity_coactivation_not_semantic_view_names() -> None:
    rng = np.random.default_rng(7)
    n = 5_000
    base = rng.normal(size=n)
    opportunity = base >= np.quantile(base, .8)
    left = rng.normal(size=n)
    right = rng.normal(size=n)
    event = rng.binomial(1, .2, size=n).astype(float)
    event[opportunity & (left > .8) & (right > .8)] = 1.
    frame = pd.DataFrame({"base": base, "event": event, "opaque_a": left, "opaque_b": right, "noise": rng.normal(size=n)})
    views, fields, edges = discover_opportunity_views(frame, ["opaque_a", "opaque_b", "noise"], base_score_column="base", label_column="event", min_joint_rows=20)
    assert views
    assert {"feature", "orientation", "activation_lift", "assigned_view"}.issubset(fields.columns)
    assert {"left", "right", "coactive_rows", "joint_synergy"}.issubset(edges.columns)
    assert any({"opaque_a", "opaque_b"}.issubset(set(features)) for features in views.values())


def test_only_train_selected_synergy_pairs_are_emitted_later() -> None:
    rng = np.random.default_rng(9)
    n = 2_000
    frame = pd.DataFrame({"base": rng.normal(size=n), "event": rng.binomial(1, .3, size=n), "a": rng.random(n), "b": rng.random(n)})
    diagnostics, train_features = opportunity_conditioned_synergy(frame, {"left": "a", "right": "b"}, base_score_column="base", label_column="event")
    later = apply_synergy_features(frame.iloc[:30], {"left": "a", "right": "b"}, diagnostics, base_score_column="base")
    expected = 2 if bool(diagnostics.selected_for_router.iloc[0]) else 0
    assert train_features.shape[1] == 2
    assert later.shape[1] == expected


def test_path_label_fields_are_never_admitted_to_specialist_views() -> None:
    columns = ["ret1h", "t2_tp3_sl2_event", "t2_tp2_sl2_event", "net_bps"]
    assert not is_permitted_feature("t2_tp3_sl2_event")
    assert permitted_causal_features(
        columns, causal_allowlist={"ret1h", "t2_tp3_sl2_event", "t2_tp2_sl2_event"}
    ) == ["ret1h"]


def test_specialist_candidate_pool_is_not_a_base_or_meta_model_input() -> None:
    from extreme_price_movements.config import CFG, SPECIALIST_CAUSAL_CONTEXT_FEATURE_KEYS

    assert SPECIALIST_CAUSAL_CONTEXT_FEATURE_KEYS
    assert "SPECIALIST_CAUSAL_CONTEXT_FEATURE_KEYS" not in CFG["base_shared_feature_keys"]
    assert "SPECIALIST_CAUSAL_CONTEXT_FEATURE_KEYS" not in CFG["meta_shared_feature_keys"]
    assert CFG["specialist_causal_context_feature_keys"] == SPECIALIST_CAUSAL_CONTEXT_FEATURE_KEYS


def test_explicit_specialist_context_can_include_perp_conversion_fields() -> None:
    """Funding/OI context is specialist-admissible without broadening base/meta."""
    from extreme_price_movements import config, pipeline_steps
    from extreme_price_movements.packb_static_point_feature_loader import (
        _provenance_backed_raw_allowlist,
    )

    expected = pipeline_steps._expected_feature_keys_from_cfg(config.CFG)
    allowlist, _, _, _ = _provenance_backed_raw_allowlist()
    for field in ("funding_z", "funding_per_hour", "oi_z", "leverage_build"):
        assert field in config.SPECIALIST_CAUSAL_CONTEXT_FEATURE_KEYS
        assert field in expected
        assert field in allowlist
