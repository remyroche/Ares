from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.regime_interaction_discovery import (
    InteractionDiscoveryConfig,
    TreeInteractionUnavailableError,
    adapt_state_probability_namespace,
    deterministic_stratified_subsample_positions,
    discover_fold_local_regime_interactions,
    exact_tree_shap_interactions,
)


class _ExactInteractionTree:
    """Small deterministic stand-in for an estimator with an exact API."""

    def shap_interaction_values(self, x: pd.DataFrame) -> np.ndarray:
        names = list(x.columns)
        out = np.zeros((len(x), len(names), len(names)), dtype=float)

        def set_pair(left: str, right: str, value: float) -> None:
            i, j = names.index(left), names.index(right)
            out[:, i, j] = value
            out[:, j, i] = value

        set_pair("signal_a", "regime_prob__trend", 4.0)
        set_pair("signal_b", "regime_prob__range", 2.0)
        set_pair("signal_b", "transition_prob__onset", 6.0)
        return out

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return (
            0.3 * x["signal_a"].to_numpy()
            + 0.5 * x["signal_b"].to_numpy()
            + 0.4 * x["regime_prob__trend"].to_numpy() * x["signal_a"].to_numpy()
            + 0.8
            * x["transition_prob__onset"].to_numpy()
            * x["signal_b"].to_numpy()
        )


class _UnsupportedTree:
    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(x), dtype=float)


def _inputs(rows: int = 120) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    index = pd.Index([f"train_{i:03d}" for i in range(rows)], name="row_id")
    x = np.linspace(-1.0, 1.0, rows)
    predictors = pd.DataFrame({"signal_a": x, "signal_b": np.sin(4 * x)}, index=index)
    target = pd.Series((x > 0).astype(float), index=index)
    regime = pd.DataFrame(
        {
            "regime_prob__trend": np.clip(0.5 + 0.4 * x, 0.05, 0.95),
            "regime_prob__range": np.clip(0.5 - 0.3 * x, 0.05, 0.95),
        },
        index=index,
    )
    transition = pd.DataFrame(
        {"transition_prob__onset": np.clip(0.4 + 0.25 * np.sin(3 * x), 0.05, 0.95)},
        index=index,
    )
    return predictors, target, regime, transition


def test_stratified_subsampling_is_deterministic_and_retains_both_classes() -> None:
    target = np.r_[np.zeros(80), np.ones(20)]
    first = deterministic_stratified_subsample_positions(target, max_rows=30, bins=4, seed=17)
    second = deterministic_stratified_subsample_positions(target, max_rows=30, bins=4, seed=17)

    assert np.array_equal(first, second)
    assert len(first) == 30
    assert set(target[first]) == {0.0, 1.0}


def test_discovers_and_keeps_regime_and_transition_candidates_separate() -> None:
    predictors, target, regime, transition = _inputs()
    config = InteractionDiscoveryConfig(
        fold_id="fold_2024_01",
        max_rows=72,
        stability_subsamples=3,
        permutation_repeats=2,
        min_probability_support=12.0,
        top_n=10,
        random_state=31,
    )
    regime_ranked, transition_ranked, metadata = discover_fold_local_regime_interactions(
        predictors,
        target,
        regime,
        transition,
        model=_ExactInteractionTree(),
        config=config,
        model_training_row_ids=predictors.index,
    )

    assert metadata["fold_id"] == "fold_2024_01"
    assert metadata["train_only_attested"] is True
    assert set(regime_ranked["namespace"]) == {"regime_probability"}
    assert set(transition_ranked["namespace"]) == {"transition_probability"}
    assert regime_ranked.iloc[0]["probability_column"] == "regime_prob__trend"
    assert regime_ranked.iloc[0]["predictor"] == "signal_a"
    assert transition_ranked.iloc[0]["probability_column"] == "transition_prob__onset"
    assert transition_ranked.iloc[0]["predictor"] == "signal_b"
    assert regime_ranked["support_rows_mean"].gt(12.0).all()
    assert regime_ranked.iloc[0]["shap_stability"] == 1.0
    assert transition_ranked["permutation_stability"].between(0.0, 1.0).all()


def test_enforces_train_only_identity_and_predictor_denylist() -> None:
    predictors, target, regime, transition = _inputs()
    config = InteractionDiscoveryConfig(fold_id="fold_a", max_rows=64, min_probability_support=8.0)

    with pytest.raises(ValueError, match="model_training_row_ids"):
        discover_fold_local_regime_interactions(
            predictors,
            target,
            regime,
            transition,
            model=_ExactInteractionTree(),
            config=config,
            model_training_row_ids=predictors.index[:-1],
        )

    predictors["target__future_net_ev"] = 1.0
    with pytest.raises(ValueError, match="forbidden"):
        discover_fold_local_regime_interactions(
            predictors,
            target,
            regime,
            transition,
            model=_ExactInteractionTree(),
            config=config,
            model_training_row_ids=predictors.index,
        )


def test_rejects_mixed_probability_namespaces() -> None:
    predictors, target, regime, transition = _inputs()
    regime["transition_prob__wrong"] = 0.5
    config = InteractionDiscoveryConfig(fold_id="fold_a")

    with pytest.raises(ValueError, match="namespace"):
        discover_fold_local_regime_interactions(
            predictors,
            target,
            regime,
            transition,
            model=_ExactInteractionTree(),
            config=config,
            model_training_row_ids=predictors.index,
        )


def test_canonical_state_probability_names_are_renamed_only_after_oof_generation() -> None:
    predictors, target, regime, transition = _inputs()
    canonical_regime = regime.rename(columns=lambda name: name.replace("regime_prob__", "regime_state_p__"))
    canonical_transition = transition.rename(columns=lambda name: name.replace("transition_prob__", "transition_state_p__"))
    adapted, source = adapt_state_probability_namespace(canonical_regime, kind="regime")

    assert source == "canonical_state_output_renamed"
    assert adapted.index.equals(canonical_regime.index)
    assert adapted.columns.tolist() == ["regime_prob__trend", "regime_prob__range"]
    _, _, metadata = discover_fold_local_regime_interactions(
        predictors,
        target,
        canonical_regime,
        canonical_transition,
        model=_ExactInteractionTree(),
        config=InteractionDiscoveryConfig(fold_id="fold_canonical", max_rows=64, min_probability_support=8.0),
        model_training_row_ids=predictors.index,
    )
    assert metadata["regime_probability_namespace_source"] == "canonical_state_output_renamed"
    assert metadata["transition_probability_namespace_source"] == "canonical_state_output_renamed"


def test_unsupported_tree_interaction_api_fails_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    # Suppress the optional TreeSHAP fallback so this test is independent of
    # whether the local environment installed the optional shap package.
    import builtins

    original_import = builtins.__import__

    def blocked_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "shap":
            raise ImportError("test without shap")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    x = pd.DataFrame({"signal_a": [0.0, 1.0]})
    with pytest.raises(TreeInteractionUnavailableError, match="exact tree-SHAP"):
        exact_tree_shap_interactions(_UnsupportedTree(), x)
