from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.causal_leaf_health import (
    CausalLeafHealthConfig,
    CausalLeafHealthError,
    build_causal_leaf_health_states,
    write_immutable_causal_leaf_health,
)


CONTRACT = "a" * 64
FAMILY = (CONTRACT, "long", "p_clear", "family_clear", "positive")
RELATION = "continuous_regime__relationship_break__trend_breadth__residual_abs_30d"


def _candidates(*, first_label: float = 1.0) -> pd.DataFrame:
    timestamps = [
        pd.Timestamp("2024-01-01 00:00:00", tz="UTC"),
        pd.Timestamp("2024-01-01 00:00:00", tz="UTC"),
        pd.Timestamp("2024-01-01 02:00:00", tz="UTC"),
        pd.Timestamp("2024-01-31 22:00:00", tz="UTC"),
        pd.Timestamp("2024-02-02 02:00:00", tz="UTC"),
    ]
    labels = [first_label, 0.0, 1.0, 0.0, 1.0]
    return pd.DataFrame({
        "candidate_id": [f"c{index}" for index in range(len(timestamps))],
        "decision_ts": timestamps,
        "feature_generation_ts": timestamps,
        "label_available_ts": [value + pd.Timedelta(hours=1) for value in timestamps],
        "side_name": ["long"] * len(timestamps),
        "head_name": ["p_clear"] * len(timestamps),
        "fold_id": ["fold_0"] * len(timestamps),
        "transport": ["transport_a"] * len(timestamps),
        "meta_partition": ["inner_oof"] * len(timestamps),
        "feature_contract_sha256": [CONTRACT] * len(timestamps),
        "semantic_label": labels,
        "head_prediction": [0.75] * len(timestamps),
        "net_bps": [75.0, -125.0, 80.0, -90.0, 70.0],
        "base_expected_bps": [10.0] * len(timestamps),
        "asset": ["A", "B", "C", "D", "E"],
    })


def _contributions(candidates: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "candidate_id": candidates["candidate_id"],
        "__ts__": candidates["decision_ts"],
        "side_name": ["long"] * len(candidates),
        "fold_id": ["fold_0"] * len(candidates),
        "head_name": ["p_clear"] * len(candidates),
        "rule_signature": ["family_clear"] * len(candidates),
        "contribution_direction": ["positive"] * len(candidates),
        "family_ensemble_tree_contribution": [0.2] * len(candidates),
    })


def _context() -> pd.DataFrame:
    timestamp = pd.date_range("2023-12-31", "2024-02-03", freq="h", tz="UTC")
    return pd.DataFrame({
        "regime_available_utc": timestamp,
        "ctx_x": np.linspace(-2.0, 2.0, len(timestamp), dtype=np.float64),
        "ctx_y": np.cos(np.linspace(0.0, 5.0, len(timestamp))),
        RELATION: np.where(np.arange(len(timestamp)) % 2, 0.30, 0.05),
    })


def _config() -> CausalLeafHealthConfig:
    return CausalLeafHealthConfig(
        min_timestamp_support=1,
        min_day_support=1,
        min_symbol_support=1,
        min_period_rows=1,
        min_periods=1,
        period_close_lag_hours=0,
        h3_min_rows=2,
        selected_context_families=frozenset((FAMILY,)),
        selected_covariance_families=frozenset((FAMILY,)),
        selected_relationship_families=frozenset((FAMILY,)),
    )


def _result(*, first_label: float = 1.0):
    candidates = _candidates(first_label=first_label)
    return build_causal_leaf_health_states(
        candidates, _contributions(candidates), causal_context=_context(),
        context_feature_columns=("ctx_x", "ctx_y", RELATION), config=_config(),
    )


def test_health_is_strictly_prior_resolved_and_same_timestamp_isolation() -> None:
    baseline = _result(first_label=1.0)
    changed = _result(first_label=0.0)
    fields = [
        "base_health__h1__p_clear__positive__posterior_correctness",
        "base_health__h1__p_clear__positive__economic_residual_bps",
    ]
    # c1 is contemporaneous with c0: neither may see c0's label.
    left = baseline.health_features.set_index("candidate_id").loc["c1", fields]
    right = changed.health_features.set_index("candidate_id").loc["c1", fields]
    np.testing.assert_allclose(left.to_numpy(float), right.to_numpy(float))
    # c2 is later than c0/c1 label availability and should now see their
    # resolved history; c0's outcome therefore changes its H1 posterior.
    left = baseline.health_features.set_index("candidate_id").loc["c2", fields]
    right = changed.health_features.set_index("candidate_id").loc["c2", fields]
    assert not np.allclose(left.to_numpy(float), right.to_numpy(float))


def test_h1_to_h5_are_token_free_finite_and_h3_refits_at_completed_boundary(tmp_path) -> None:
    result = _result()
    assert not any("leaf_" in name for name in result.health_features)
    fields = [name for name in result.health_features if name.startswith("base_health__")]
    assert fields
    assert np.isfinite(result.health_features[fields].to_numpy(float)).all()
    february = result.health_features.set_index("candidate_id").loc["c4"]
    assert february["base_health__h3__p_clear__positive__availability"] == pytest.approx(1.0)
    assert february["base_health__h5__p_clear__positive__availability"] == pytest.approx(1.0)
    assert not result.period_metrics.empty
    assert not result.portability_scores.empty
    output = write_immutable_causal_leaf_health(result, tmp_path / "health")
    assert (output / "base_leaf_health_features_oof.parquet").is_file()
    assert (output / "leaf_covariance_reference_manifest.json").is_file()
    with pytest.raises(FileExistsError):
        write_immutable_causal_leaf_health(result, output)


def test_rejects_raw_leaf_identifiers_and_late_context() -> None:
    candidates = _candidates()
    contributions = _contributions(candidates).assign(leaf_token=1)
    with pytest.raises(CausalLeafHealthError, match="raw local leaf"):
        build_causal_leaf_health_states(candidates, contributions, config=_config())
    late = _context().assign(regime_available_utc=lambda frame: frame["regime_available_utc"] + pd.Timedelta(days=365))
    with pytest.raises(CausalLeafHealthError, match="prior-available"):
        build_causal_leaf_health_states(
            candidates, _contributions(candidates), causal_context=late,
            context_feature_columns=("ctx_x", "ctx_y", RELATION), config=_config(),
        )


def test_multiple_semantic_heads_collapse_to_one_candidate_health_row() -> None:
    base = _candidates().iloc[:2].copy()
    candidate_parts: list[pd.DataFrame] = []
    contribution_parts: list[pd.DataFrame] = []
    for head in ("p_adverse", "p_weak", "p_clear"):
        candidate = base.copy()
        candidate["head_name"] = head
        candidate_parts.append(candidate)
        contribution = _contributions(candidate)
        contribution["head_name"] = head
        contribution["rule_signature"] = f"{head}_family"
        contribution_parts.append(contribution)
    result = build_causal_leaf_health_states(
        pd.concat(candidate_parts, ignore_index=True),
        pd.concat(contribution_parts, ignore_index=True),
        causal_context=_context(),
        context_feature_columns=("ctx_x", "ctx_y", RELATION),
        config=CausalLeafHealthConfig(
            min_timestamp_support=1, min_day_support=1, min_symbol_support=1,
        ),
    )
    assert len(result.health_features) == 2
    assert result.health_features["candidate_id"].is_unique
    assert "base_health__h1__p_adverse__positive__posterior_correctness" in result.health_features
    assert "base_health__h1__p_weak__positive__posterior_correctness" in result.health_features


def test_predecessor_selection_activation_zeroes_h3_h4_h5_before_its_cutoff() -> None:
    candidates = _candidates()
    config = replace(
        _config(), family_selection_effective_utc="2024-02-01T00:00:00Z",
    )
    result = build_causal_leaf_health_states(
        candidates, _contributions(candidates), causal_context=_context(),
        context_feature_columns=("ctx_x", "ctx_y", RELATION), config=config,
    )
    values = result.health_features.set_index("candidate_id")
    fields = (
        "base_health__h3__p_clear__positive__availability",
        "base_health__h4__p_clear__positive__availability",
        "base_health__h5__p_clear__positive__availability",
    )
    assert (values.loc[["c0", "c1", "c2", "c3"], list(fields)] == 0.0).all().all()
    assert values.loc["c4", "base_health__h3__p_clear__positive__availability"] == 1.0
    assert values.loc["c4", "base_health__h5__p_clear__positive__availability"] == 1.0
