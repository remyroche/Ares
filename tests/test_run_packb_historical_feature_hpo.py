from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/run_packb_historical_feature_hpo.py"
SPEC = importlib.util.spec_from_file_location("packb_hist_hpo", SCRIPT)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_approved_historical_contract_is_exact() -> None:
    features, evidence = runner._historical_features(
        runner.DEFAULT_HISTORICAL_CONTRACT,
        runner.DEFAULT_HISTORICAL_PROCESS,
    )
    assert len(features["long"]) == 55
    assert len(features["short"]) == 37
    assert evidence["selection_fold"] == "2026-05-31_2026-06-30"
    assert evidence["exception_scope"] == "feature_names_only"


def test_selector_requires_every_approved_feature_to_pass_coverage() -> None:
    selector = runner.ApprovedHistoricalSelector(
        side="long",
        expected_features=("a", "b"),
        exception_evidence={"selection_fold": "fixed"},
    )
    value = type(
        "Selection",
        (),
        {"side": "long", "candidate_features": ("a",)},
    )()
    with pytest.raises(
        runner.HistoricalFeatureHPORunnerError, match="coverage admission"
    ):
        selector(value)


def test_composite_source_partition_is_disjoint_and_ordered() -> None:
    class Candidate:
        def load(self, ledger, requested):
            return pd.DataFrame({name: [1.0] for name in requested})

    class Representation:
        def __call__(self, ledger, requested):
            return pd.DataFrame({name: [3.0] for name in requested})

    class Guard:
        def checkpoint(self, _name):
            return None

    loader = runner.HistoricalCompositeFeatureLoader(
        side="long",
        all_features=("candidate", "dae_b16_06"),
        candidate_features=("candidate",),
        candidate_loader=Candidate(),
        representation_loader=Representation(),
        generated_features=("dae_b16_06",),
        feature_store=Path("/unused"),
        resource_guard=Guard(),
    )
    ledger = pd.DataFrame({"candidate_id": ["x"]})
    result = loader(ledger, ("dae_b16_06", "candidate"))
    assert list(result.columns) == ["dae_b16_06", "candidate"]
    assert result.iloc[0].tolist() == [3.0, 1.0]
    assert loader.source_contract()["source_precedence"].startswith("candidate")


def test_cached_representation_loader_preserves_requested_order() -> None:
    ledger = pd.DataFrame({"candidate_id": ["a", "b"]})
    values = pd.DataFrame({"dae_b16_06": [1.0, 2.0], "gmm_ood_score": [3.0, 4.0]})
    loader = runner.CachedRepresentationFeatureLoader(ledger, values)
    result = loader(
        pd.DataFrame({"candidate_id": ["b", "a"]}),
        ("gmm_ood_score", "dae_b16_06"),
    )
    assert list(result.columns) == ["gmm_ood_score", "dae_b16_06"]
    assert result.to_numpy().tolist() == [[4.0, 2.0], [3.0, 1.0]]
