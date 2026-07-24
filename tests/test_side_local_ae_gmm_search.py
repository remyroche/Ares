import numpy as np
import pandas as pd

from extreme_price_movements.side_local_ae_gmm_search import (
    available_layer_features,
    config_feature_universe,
    correlation_cluster_representatives,
    empty_state_block,
    nested_feature_prefixes,
    stable_feature_filter,
)


def test_meta_feature_universe_admits_observable_base_context_not_outcomes():
    frame = pd.DataFrame(
        {
            "ret1h": [0.1, 0.2],
            "base_score_oof": [0.3, 0.4],
            "base_uncertainty": [0.1, 0.2],
            "realized_exec_margin": [0.5, -0.2],
            "target_soft": [0.4, 0.6],
        }
    )
    names = available_layer_features(frame, "meta")
    assert "base_score_oof" in names
    assert "base_uncertainty" in names
    assert "realized_exec_margin" not in names
    assert "target_soft" not in names
    assert "ret1h" in config_feature_universe("base")


def test_correlation_groups_keep_most_relevant_stable_representative():
    rng = np.random.default_rng(3)
    x = rng.normal(size=400)
    frame = pd.DataFrame({"a": x, "b": x + rng.normal(0, 1e-4, 400), "c": rng.normal(size=400)})
    stats = stable_feature_filter(frame, ["a", "b", "c"])
    selected, membership = correlation_cluster_representatives(frame, stats, {"a": 0.1, "b": 0.8, "c": 0.2})
    assert "b" in selected
    assert "a" not in selected
    assert membership["representative"].sum() == len(selected)


def test_prefixes_are_nested_and_state_outputs_are_side_qualified():
    rank = pd.DataFrame(
        {"feature": [f"f{i}" for i in range(160)], "mda_stable_importance": np.arange(160, 0, -1)}
    )
    prefixes = nested_feature_prefixes(rank)
    assert prefixes[30] == prefixes[75][:30]
    assert prefixes[75] == prefixes[150][:75]
    block = empty_state_block(pd.RangeIndex(2), "meta_short_state", components=3, latent_dim=12)
    assert "meta_short_state_posterior_00" in block
    assert "meta_short_state_latent_11" in block
    assert "meta_short_state_active" in block
    assert (block["meta_short_state_component_id_local"] == -1).all()
