from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCOPE = ROOT / "configs/root_cause_two_head_scope_20260731_v1.json"


def _scope() -> dict:
    return json.loads(SCOPE.read_text(encoding="utf-8"))


def test_root_cause_scope_keeps_exactly_base_and_residual_layers() -> None:
    scope = _scope()
    assert [row["id"] for row in scope["trainable_layers"]] == [
        "base_directional_alpha",
        "stopped_gradient_residual",
    ]
    assert scope["trainable_layers"][1]["requires_oof_base_prediction"] is True


def test_root_cause_scope_disables_every_other_head() -> None:
    disabled = set(_scope()["disabled_layers"])
    assert {
        "catboost_path_archetype",
        "path_auxiliary_heads",
        "execution_ev_meta_head",
        "entry_timing_meta_head",
        "action_policy_head",
        "sizing_head",
        "portfolio_optimizer",
    } <= disabled


def test_directional_and_residual_metrics_have_separate_namespaces() -> None:
    scope = _scope()
    layers = scope["trainable_layers"]
    assert layers[0]["metric_namespace"] == "base_directional__"
    assert layers[1]["metric_namespace"] == "residual_economic__"
    assert scope["reporting_contract"]["mixing_base_and_residual_metric_rows"] is False


def test_root_cause_scope_is_research_only_without_portfolio_changes() -> None:
    scope = _scope()
    assert scope["research_only"] is True
    assert scope["promotion_allowed"] is False
    assert scope["portfolio_changes_allowed"] is False
