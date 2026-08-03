from pathlib import Path

import pandas as pd

from scripts.audit_recurring_transition_taxonomy_stability import (
    morphology_classifier_agreement,
    stable_transition_metrics,
    summarize_morphology,
    tool_inventory,
)


def _morphology() -> pd.DataFrame:
    return pd.DataFrame({
        "event_id": ["a", "b", "c", "d"],
        "anchor_source_utc": pd.to_datetime(["2022-01-01", "2023-01-01", "2024-01-01", "2024-02-01"], utc=True),
        "source_state": [0, 0, 1, 1], "destination_state": [1, 1, 0, 0],
        "morphology__posterior_m00": [.8, .9, .1, .2], "morphology__posterior_m01": [.2, .1, .9, .8],
        "morphology__entropy": [.5, .3, .4, .5], "morphology__top2_margin": [.6, .8, .8, .6],
        "morphology__component_id": ["m00", "m00", "m01", "m01"], "morphology__abstained": [0, 0, 0, 1], "oof_fold": [0, 0, 1, 1],
    })


def test_morphology_summary_does_not_equate_fold_local_ids() -> None:
    output = summarize_morphology(_morphology())
    assert set(output["cross_fold_alignment"]) == {"NOT_IDENTIFIABLE_COMPONENT_IDS_ARE_FOLD_LOCAL"}
    assert output.loc[output["fold_local_component"].eq("m00"), "recurs_across_eras"].iloc[0]


def test_morphology_classifier_agreement_and_abstention() -> None:
    morphology = _morphology()
    classifier = morphology.loc[:, ["event_id", "anchor_source_utc", "oof_fold"]].copy()
    classifier["classifier__p_m00"] = [.9, .9, .1, .1]
    classifier["classifier__p_m01"] = [.1, .1, .9, .9]
    output = morphology_classifier_agreement(morphology, classifier)
    assert output.loc[output["slice"].eq("all_events"), "agreement"].iloc[0] == 1.0
    assert output.loc[output["slice"].eq("non_abstained"), "events"].iloc[0] == 3


def test_stable_metrics_use_oof_probabilities() -> None:
    frame = pd.DataFrame({
        "event_id": ["a", "b", "c", "d"],
        "anchor_source_utc": pd.to_datetime(["2022-01-01", "2022-01-02", "2023-01-01", "2023-01-02"], utc=True),
        "target__stable_vs_transition": [0, 1, 0, 1], "classifier__p_1": [.1, .9, .2, .8], "oof_fold": [0, 0, 1, 1],
    })
    metrics, reliability = stable_transition_metrics(frame)
    assert metrics.iloc[0].roc_auc == 1.0
    assert not reliability.empty


def test_inventory_marks_native_brl_as_executable_without_optional_dependency(tmp_path: Path) -> None:
    output = tool_inventory(tmp_path, tmp_path)
    row = output.loc[output.tool.eq("Bayesian_Rule_List")].iloc[0]
    assert row.status == "implemented_only"
    assert row.dependency_executable
