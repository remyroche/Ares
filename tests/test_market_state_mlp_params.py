from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.run_meta_market_state_encoder_ablation import (
    DEFAULT_MLP_PARAMS,
    _load_mlp_params,
    _monthly_rank_contract_metrics,
    _causal_ev_residual,
    _same_identity_rows,
    _select_ev_features,
    _write_mlp_params,
)


def test_identity_comparison_ignores_categorical_dtype_only() -> None:
    timestamp = pd.to_datetime(["2026-04-01T00:00:00Z"], utc=True)
    left = pd.DataFrame(
        {
            "__ts__": timestamp,
            "__symbol__": pd.Series(["BTC"], dtype="category"),
            "side_name": pd.Series(["long"], dtype="category"),
            "archetype_policy_key": pd.Series(["a"], dtype="category"),
        }
    )
    right = left.astype(
        {"__symbol__": "string", "side_name": "string", "archetype_policy_key": "string"}
    )

    assert _same_identity_rows(left, right)
    right.loc[0, "__symbol__"] = "ETH"
    assert not _same_identity_rows(left, right)


def test_predecessor_references_are_frozen_before_context_projection() -> None:
    import scripts.run_meta_market_state_encoder_ablation as module

    source = Path(module.__file__).read_text()
    catalog_pos = source.index("catalog = _feature_catalog(history)")
    references_pos = source.index(
        "predecessor_references = _fit_references(history, catalog, 1)",
        catalog_pos,
    )
    context_pos = source.index("history = _merge_observable_context", catalog_pos)
    export_pos = source.index("_export_composite_policy(", context_pos)

    assert catalog_pos < references_pos < context_pos < export_pos
    assert source.count(
        "predecessor_references = _fit_references(history, catalog, 1)"
    ) == 1


def test_mlp_hpo_is_optional_with_stable_defaults() -> None:
    params, source = _load_mlp_params(None)

    assert source == "checked_in_default"
    assert params == DEFAULT_MLP_PARAMS
    assert params is not DEFAULT_MLP_PARAMS


def test_promoted_mlp_params_override_defaults(tmp_path) -> None:
    path = tmp_path / "best_mlp_params.json"
    path.write_text(
        json.dumps(
            {
                "schema": "market_state_mlp_params_v1",
                "mlp_params": {
                    "hidden_layer_sizes": [48, 24, 12],
                    "alpha": 0.4,
                },
            }
        )
    )

    params, source = _load_mlp_params(path)

    assert source == str(path)
    assert params["hidden_layer_sizes"] == (48, 24, 12)
    assert params["alpha"] == 0.4
    assert params["noise_std"] == DEFAULT_MLP_PARAMS["noise_std"]


def test_mlp_params_round_trip(tmp_path) -> None:
    path = tmp_path / "best_mlp_params.json"
    _write_mlp_params(path, DEFAULT_MLP_PARAMS, source="chronological_mlp_hpo")

    params, source = _load_mlp_params(path)

    assert source == str(path)
    assert params == DEFAULT_MLP_PARAMS


def test_causal_ev_residual_does_not_use_later_blocks() -> None:
    n = 1_600
    frame = pd.DataFrame(
        {
            "ev_after_1pct": np.linspace(-0.01, 0.02, n, dtype=np.float32),
            "policy_parent_rank": np.linspace(0, 1, n, dtype=np.float32),
            "hit_probability": np.linspace(0.2, 0.8, n, dtype=np.float32),
        }
    )
    baseline = _causal_ev_residual(frame)
    changed = frame.copy()
    changed.loc[1_200:, "ev_after_1pct"] += 10.0
    revised = _causal_ev_residual(changed)

    assert np.isnan(baseline[:400]).all()
    np.testing.assert_allclose(baseline[400:1_200], revised[400:1_200])


def test_pre_mlp_selection_uses_automatic_count_and_reports_scores() -> None:
    rng = np.random.default_rng(9)
    n = 2_000
    signal = rng.normal(size=n).astype(np.float32)
    rank = rng.uniform(size=n).astype(np.float32)
    frame = pd.DataFrame(
        {
            "policy_parent_rank": rank,
            "hit_probability": np.clip(rank + rng.normal(0, 0.1, n), 0, 1),
            "ev_after_1pct": 0.01 * signal + rng.normal(0, 0.005, n),
            **{
                f"feature_{i}": signal + rng.normal(0, 0.1 + i * 0.05, n)
                for i in range(30)
            },
        }
    )
    selected, report = _select_ev_features(
        frame, [f"feature_{i}" for i in range(30)], max_features=None, seed=11
    )

    assert 4 <= len(selected) <= 48
    assert report["automatic_feature_cap"].nunique() == 1
    assert {
        "conditional_oof_gain",
        "weighted_binned_mi",
        "interaction_constituent_gain",
        "selected",
    }.issubset(report.columns)


def test_monthly_rank_contract_uses_parent_activity_per_month() -> None:
    frame = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(
                [
                    "2026-04-01T00:00:00Z",
                    "2026-04-02T00:00:00Z",
                    "2026-05-01T00:00:00Z",
                    "2026-05-02T00:00:00Z",
                ],
                utc=True,
            ),
            "__fold__": [2, 2, 3, 3],
            "policy_parent_rank": [0.91, 0.10, 0.20, 0.92],
            "rank_mlp_direct": [0.10, 0.90, 0.95, 0.05],
            "expected_ev_rank_score": [0.20, 0.80, 0.90, 0.10],
            "ev_after_1pct": [0.01, 0.02, 0.03, -0.01],
            "clean_exec": [1, 1, 1, 0],
            "dirty_positive": [0, 0, 0, 1],
            "full_path_bad_mae_1r": [0, 0, 0, 1],
            "timeout": [0, 0, 0, 1],
        }
    )

    report = _monthly_rank_contract_metrics(frame)

    assert report["target_activity"].eq(1).all()
    assert report["selected_rows"].eq(1).all()
    assert report["missing_score_rows"].eq(0).all()
    april_mlp = report.loc[
        report["month"].eq("2026-04") & report["arm"].eq("mlp_direct")
    ].iloc[0]
    assert april_mlp["mean_net_ev_after_1pct"] == pytest.approx(0.02)
