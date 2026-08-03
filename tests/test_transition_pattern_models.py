import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.transition_pattern_models import (
    BayesianRuleListChallenger,
    MorphologyGateConfig,
    TransitionClassifierAdapter,
    TransitionMorphologyConfig,
    TransitionMorphologyEmbedder,
    component_support_table,
    minimum_recurrence_stability_gate,
    validate_preonset_sequence_columns,
)


def _events(rows: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    cluster = np.repeat([0, 1], repeats=rows // 2)
    return pd.DataFrame(
        {
            "event_id": [f"event_{index}" for index in range(rows)],
            "sequence_available_utc": pd.date_range("2025-01-01", periods=rows, freq="D", tz="UTC"),
            "era": np.where(np.arange(rows) < rows // 3, "2024", np.where(np.arange(rows) < 2 * rows // 3, "2025", "2026")),
            "source_state": cluster,
            "destination_state": 1 - cluster,
            "sequence__breadth__mean_12h": cluster * 5.0 + rng.normal(0, 0.15, rows),
            "sequence__breadth__slope_per_hour_12h": cluster * 0.8 + rng.normal(0, 0.05, rows),
            "sequence__volatility__mean_12h": (1 - cluster) * 3.0 + rng.normal(0, 0.10, rows),
            "target__stable_vs_transition": cluster,
            "morphology_label": np.where(cluster == 0, "washout", "covering"),
        }
    )


def test_train_only_embedding_emits_simplex_and_rejects_state_or_post_onset_inputs() -> None:
    train = _events()
    columns = [
        "sequence__breadth__mean_12h",
        "sequence__breadth__slope_per_hour_12h",
        "sequence__volatility__mean_12h",
    ]
    model = TransitionMorphologyEmbedder(
        TransitionMorphologyConfig(n_components=2, embedding_components=2, min_component_events=3, min_component_eras=1)
    ).fit(train, feature_columns=columns, era_column="era")
    assert np.allclose(model.imputer.statistics_, train[columns].median().to_numpy())
    test = train.iloc[:4].copy()
    test.loc[:, columns] = 1_000_000.0  # must not refit preprocessing on score rows
    output = model.transform(test)
    posterior = output.filter(like="morphology__posterior_").to_numpy()
    assert np.allclose(posterior.sum(axis=1), 1.0)
    assert model.feature_columns == columns
    with pytest.raises(ValueError, match="non-causal"):
        validate_preonset_sequence_columns(train, ["destination_state"])
    with pytest.raises(ValueError, match="non-causal"):
        validate_preonset_sequence_columns(train.assign(sequence__future_mfe__mean_12h=1.0), ["sequence__future_mfe__mean_12h"])


def test_unsupported_morphologies_abstain_but_keep_soft_simplex() -> None:
    train = _events()
    columns = ["sequence__breadth__mean_12h", "sequence__volatility__mean_12h"]
    model = TransitionMorphologyEmbedder(
        TransitionMorphologyConfig(n_components=2, embedding_components=2, min_component_events=100, min_component_eras=1)
    ).fit(train, feature_columns=columns, era_column="era")
    output = model.transform(train)
    assert output["morphology__abstained"].eq(1).all()
    assert output["morphology__component_id"].eq("abstain").all()
    assert np.allclose(output.filter(like="morphology__posterior_").sum(axis=1), 1.0)


def test_classifier_adapters_and_recurrence_stability_gate() -> None:
    train = _events()
    columns = ["sequence__breadth__mean_12h", "sequence__breadth__slope_per_hour_12h", "sequence__volatility__mean_12h"]
    binary = TransitionClassifierAdapter(n_estimators=8).fit(
        train,
        target_column="target__stable_vs_transition",
        feature_columns=columns,
    )
    probability = binary.predict_proba(train.iloc[:5])
    assert np.allclose(probability.sum(axis=1), 1.0)
    assert binary.backend in {"lightgbm", "hist_gradient_boosting"}
    multi = TransitionClassifierAdapter(n_estimators=8).fit(
        train,
        target_column="morphology_label",
        feature_columns=columns,
    )
    assert set(multi.predict_proba(train.iloc[:3]).columns) == {"classifier__p_covering", "classifier__p_washout"}
    brl = BayesianRuleListChallenger()
    assert brl.status == "unfitted"
    brl.fit(train, target_column="target__stable_vs_transition", feature_columns=columns)
    probability = brl.predict_proba(train)
    assert brl.status == "fitted"
    assert brl.backend in {"imodels_mcmc_brl", "native_beta_binomial_map"}
    assert probability.between(0.0, 1.0).all()
    if not brl.imodels_available:
        assert brl.backend == "native_beta_binomial_map"
        description = brl.describe()
        assert description and description[-1]["binary_features"] == ["DEFAULT"]

    support = component_support_table(["m00"] * 8 + ["m01"] * 2, era=["2024"] * 4 + ["2025"] * 4 + ["2024"] * 2, min_events=3, min_eras=2)
    assert support.loc[support["morphology_component_id"].eq("m00"), "support_pass"].item()
    assert not support.loc[support["morphology_component_id"].eq("m01"), "support_pass"].item()
    gate = minimum_recurrence_stability_gate(
        events=8,
        eras=3,
        bootstrap_ari=0.65,
        posterior_correlation=0.75,
        config=MorphologyGateConfig(),
    )
    assert gate["retained"]
    assert not minimum_recurrence_stability_gate(
        events=8,
        eras=1,
        bootstrap_ari=0.9,
        posterior_correlation=0.9,
        config=MorphologyGateConfig(),
    )["retained"]
