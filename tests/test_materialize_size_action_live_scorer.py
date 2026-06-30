import json
from pathlib import Path

import pandas as pd

from scripts import materialize_size_action_live_scorer as materializer


ARM = "C3fo_bagged_safety_c3ed_or_calibrated_group_hurdle_positive_value_acceptance_union_gate"


def test_parse_head_decision_thresholds_supports_cli_and_json() -> None:
    parsed = materializer._parse_head_decision_thresholds(
        "short_asset:p_intervene_min=0.8,pred_delta_J_min=320;short_boll:p_intervene_min=0.2"
    )

    assert parsed == {
        "short_asset": {"p_intervene_min": 0.8, "pred_delta_J_min": 320.0},
        "short_boll": {"p_intervene_min": 0.2},
    }

    parsed_json = materializer._parse_head_decision_thresholds(
        '{"short_asset": {"positive_value_min": 0.7}}'
    )

    assert parsed_json == {"short_asset": {"positive_value_min": 0.7}}


def test_materialize_full_arm_scorer_writes_fail_closed_bundle(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "research_run"
    run_dir.mkdir()
    pd.DataFrame(
        [
            {
                "timestamp": "2026-05-01 00:00:00+00:00",
                "strategy_id": "short_asset",
                "split": "train",
                "multiplier": 0.0,
            }
        ]
    ).to_csv(run_dir / "size_action_exact_panel.csv", index=False)
    (run_dir / "manifest.json").write_text(json.dumps({"policy_variant": "refit_bar4_strategy_bar2"}))
    freeze = tmp_path / "size_action_freeze_manifest.json"
    freeze.write_text(
        json.dumps(
            {
                "arm": ARM,
                "run_dir": str(run_dir),
                "source_manifest": {"policy_variant": "refit_bar4_strategy_bar2"},
            }
        )
    )

    def fake_fit(panel, *, run_dir, arm, max_features, seed, fit_split):
        assert len(panel) == 1
        components = {
            "stage1_intervention_classifier": {
                "model": {"kind": "binary_classifier", "constant": 0.8},
                "feature_columns": ["strategy_rank_q90"],
            },
            "action_selector": {
                "model": {"kind": "binary_classifier", "constant": 0.9},
                "feature_columns": ["strategy_rank_q90", "wallet"],
            },
            "full_value_regressor": {
                "model": {"kind": "regressor", "constant": 12.0},
                "feature_columns": ["strategy_rank_q90", "wallet"],
            },
            "immediate_value_regressor": {
                "model": {"kind": "regressor", "constant": 4.0},
                "feature_columns": ["strategy_rank_q90", "wallet"],
            },
            "capacity_value_regressor": {
                "model": {"kind": "regressor", "constant": 8.0},
                "feature_columns": ["strategy_rank_q90", "wallet"],
            },
            "positive_value_classifier": {
                "model": {"kind": "binary_classifier", "constant": 0.95},
                "feature_columns": ["strategy_rank_q90", "wallet"],
            },
        }
        return (
            components,
            ["strategy_rank_q90", "wallet"],
            {
                "stage1_intervention_classifier": pd.Series({"strategy_rank_q90": 0.7, "wallet": 100.0}),
                "action_selector": pd.Series({"strategy_rank_q90": 0.7, "wallet": 100.0}),
                "full_value_regressor": pd.Series({"strategy_rank_q90": 0.7, "wallet": 100.0}),
                "immediate_value_regressor": pd.Series({"strategy_rank_q90": 0.7, "wallet": 100.0}),
                "capacity_value_regressor": pd.Series({"strategy_rank_q90": 0.7, "wallet": 100.0}),
                "positive_value_classifier": pd.Series({"strategy_rank_q90": 0.7, "wallet": 100.0}),
            },
            {"short_asset": 0},
            {"fit_rows": 42, "positive_rows": 7},
        )

    monkeypatch.setattr(materializer, "_fit_full_arm_scorer", fake_fit)
    out_dir = tmp_path / "live_scorer"

    payload = materializer.materialize_bundle(
        freeze_manifest_path=freeze,
        run_dir=None,
        out_dir=out_dir,
        arm=ARM,
        material_gain=50.0,
        top_fraction=0.075,
        max_features=96,
        seed=1729,
        fit_split="train",
        min_head_group_rows=1,
    )

    assert payload["coverage"] == "full_arm"
    assert payload["missing_components"] == []
    assert payload["fail_closed"] is True
    assert payload["feature_columns"] == ["strategy_rank_q90", "wallet"]
    assert (out_dir / "size_action_live_scorer.joblib").exists()
    assert (out_dir / "size_action_live_feature_contract.json").exists()
    assert (out_dir / "size_action_live_imputation.json").exists()
    assert (out_dir / "size_action_live_policy_contract.json").exists()
    assert (out_dir / "size_action_live_scorer_manifest.json").exists()

    policy = json.loads((out_dir / "size_action_live_policy_contract.json").read_text())
    assert policy["coverage"] == "full_arm"
    assert policy["missing_component_blocker"] is None
    assert policy["head_specific_enabled"] is True
    assert policy["head_specific_heads"] == ["short_asset"]

    scored = materializer.score_size_action_frame(
        out_dir,
        pd.DataFrame(
            [
                {
                    "timestamp": "2026-05-01 00:00:00+00:00",
                    "strategy_id": "short_asset",
                    "multiplier": 1.0,
                    "strategy_rank_q90": 0.8,
                    "wallet": 100.0,
                    "action_binds": 0.0,
                },
                {
                    "timestamp": "2026-05-01 00:00:00+00:00",
                    "strategy_id": "short_asset",
                    "multiplier": 0.5,
                    "strategy_rank_q90": 0.8,
                    "wallet": 100.0,
                    "action_binds": 1.0,
                },
            ]
        ),
    )
    assert bool(scored.loc[0, "accepted"]) is True
    assert scored.loc[0, "selected_multiplier"] == 0.5
    assert scored.loc[0, "pred_delta_J"] == 12.0

    rejected = materializer.score_size_action_frame(
        out_dir,
        pd.DataFrame(
            [
                {
                    "timestamp": "2026-05-01 01:00:00+00:00",
                    "strategy_id": "short_asset",
                    "multiplier": 0.5,
                    "strategy_rank_q90": 0.8,
                    "action_binds": 1.0,
                },
            ]
        ),
    )
    assert bool(rejected.loc[0, "accepted"]) is False
    assert rejected.loc[0, "selected_multiplier"] == 1.0
    assert "missing:wallet" in rejected.loc[0, "reject_reason"]


def test_score_uses_head_specific_component_stack(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "research_run"
    run_dir.mkdir()
    pd.DataFrame(
        [
            {
                "timestamp": "2026-05-01 00:00:00+00:00",
                "strategy_id": "short_asset",
                "split": "train",
                "multiplier": 0.0,
            },
            {
                "timestamp": "2026-05-01 00:00:00+00:00",
                "strategy_id": "long_bars",
                "split": "train",
                "multiplier": 0.0,
            },
        ]
    ).to_csv(run_dir / "size_action_exact_panel.csv", index=False)
    (run_dir / "manifest.json").write_text(json.dumps({"policy_variant": "refit_bar4_strategy_bar2"}))
    freeze = tmp_path / "size_action_freeze_manifest.json"
    freeze.write_text(json.dumps({"arm": ARM, "run_dir": str(run_dir)}))

    def fake_fit(panel, *, run_dir, arm, max_features, seed, fit_split):
        heads = {materializer._strategy_head(x) for x in panel["strategy_id"].astype(str)}
        head = next(iter(heads)) if len(heads) == 1 else "global"
        full_value = {"short_asset": 21.0, "long_bars": 7.0}.get(head, 99.0)
        components = {
            "stage1_intervention_classifier": {
                "model": {"kind": "binary_classifier", "constant": 0.8},
                "feature_columns": ["wallet"],
            },
            "action_selector": {
                "model": {"kind": "binary_classifier", "constant": 0.9},
                "feature_columns": ["wallet"],
            },
            "full_value_regressor": {
                "model": {"kind": "regressor", "constant": full_value},
                "feature_columns": ["wallet"],
            },
            "immediate_value_regressor": {
                "model": {"kind": "regressor", "constant": 1.0},
                "feature_columns": ["wallet"],
            },
            "capacity_value_regressor": {
                "model": {"kind": "regressor", "constant": 2.0},
                "feature_columns": ["wallet"],
            },
            "positive_value_classifier": {
                "model": {"kind": "binary_classifier", "constant": 0.95},
                "feature_columns": ["wallet"],
            },
        }
        medians = {name: pd.Series({"wallet": 100.0}) for name in components}
        strategy_map = {str(strategy): idx for idx, strategy in enumerate(sorted(panel["strategy_id"].astype(str).unique()))}
        return components, ["wallet"], medians, strategy_map, {"fit_rows": len(panel), "scope": head}

    monkeypatch.setattr(materializer, "_fit_full_arm_scorer", fake_fit)
    out_dir = tmp_path / "live_scorer"
    materializer.materialize_bundle(
        freeze_manifest_path=freeze,
        run_dir=None,
        out_dir=out_dir,
        arm=ARM,
        material_gain=50.0,
        top_fraction=0.075,
        max_features=96,
        seed=1729,
        fit_split="train",
        min_head_group_rows=1,
    )

    scored = materializer.score_size_action_frame(
        out_dir,
        pd.DataFrame(
            [
                {
                    "timestamp": "2026-05-01 01:00:00+00:00",
                    "strategy_id": "short_asset",
                    "multiplier": 0.5,
                    "wallet": 100.0,
                    "action_binds": 1.0,
                },
                {
                    "timestamp": "2026-05-01 01:00:00+00:00",
                    "strategy_id": "long_bars",
                    "multiplier": 0.5,
                    "wallet": 100.0,
                    "action_binds": 1.0,
                },
            ]
        ),
    )

    by_strategy = scored.set_index("strategy_id")
    assert by_strategy.loc["short_asset", "component_scope"] == "head:short_asset"
    assert by_strategy.loc["long_bars", "component_scope"] == "head:long_bars"
    assert bool(by_strategy.loc["short_asset", "head_specific_component"]) is True
    assert by_strategy.loc["short_asset", "pred_delta_J"] == 21.0
    assert by_strategy.loc["long_bars", "pred_delta_J"] == 7.0


def test_score_applies_head_specific_decision_thresholds(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "research_run"
    run_dir.mkdir()
    pd.DataFrame(
        [
            {
                "timestamp": "2026-05-01 00:00:00+00:00",
                "strategy_id": "short_asset",
                "split": "train",
                "multiplier": 0.0,
            },
            {
                "timestamp": "2026-05-01 00:00:00+00:00",
                "strategy_id": "short_boll",
                "split": "train",
                "multiplier": 0.0,
            },
        ]
    ).to_csv(run_dir / "size_action_exact_panel.csv", index=False)
    (run_dir / "manifest.json").write_text(json.dumps({"policy_variant": "refit_bar4_strategy_bar2"}))
    freeze = tmp_path / "size_action_freeze_manifest.json"
    freeze.write_text(json.dumps({"arm": ARM, "run_dir": str(run_dir)}))

    def fake_fit(panel, *, run_dir, arm, max_features, seed, fit_split):
        heads = {materializer._strategy_head(x) for x in panel["strategy_id"].astype(str)}
        head = next(iter(heads)) if len(heads) == 1 else "global"
        full_value = {"short_asset": 21.0, "short_boll": 21.0}.get(head, 99.0)
        components = {
            "stage1_intervention_classifier": {
                "model": {"kind": "binary_classifier", "constant": 0.8},
                "feature_columns": ["wallet"],
            },
            "action_selector": {
                "model": {"kind": "binary_classifier", "constant": 0.9},
                "feature_columns": ["wallet"],
            },
            "full_value_regressor": {
                "model": {"kind": "regressor", "constant": full_value},
                "feature_columns": ["wallet"],
            },
            "immediate_value_regressor": {
                "model": {"kind": "regressor", "constant": 1.0},
                "feature_columns": ["wallet"],
            },
            "capacity_value_regressor": {
                "model": {"kind": "regressor", "constant": 2.0},
                "feature_columns": ["wallet"],
            },
            "positive_value_classifier": {
                "model": {"kind": "binary_classifier", "constant": 0.95},
                "feature_columns": ["wallet"],
            },
        }
        medians = {name: pd.Series({"wallet": 100.0}) for name in components}
        strategy_map = {str(strategy): idx for idx, strategy in enumerate(sorted(panel["strategy_id"].astype(str).unique()))}
        return components, ["wallet"], medians, strategy_map, {"fit_rows": len(panel), "scope": head}

    monkeypatch.setattr(materializer, "_fit_full_arm_scorer", fake_fit)
    out_dir = tmp_path / "live_scorer"
    materializer.materialize_bundle(
        freeze_manifest_path=freeze,
        run_dir=None,
        out_dir=out_dir,
        arm=ARM,
        material_gain=50.0,
        top_fraction=0.075,
        max_features=96,
        seed=1729,
        fit_split="train",
        min_head_group_rows=1,
        head_decision_thresholds={
            "short_asset": {"pred_delta_J_min": 30.0},
            "short_boll": {"pred_delta_J_min": 10.0},
        },
    )

    policy = json.loads((out_dir / "size_action_live_policy_contract.json").read_text())
    assert policy["head_decision_thresholds"]["short_asset"]["pred_delta_J_min"] == 30.0

    scored = materializer.score_size_action_frame(
        out_dir,
        pd.DataFrame(
            [
                {
                    "timestamp": "2026-05-01 01:00:00+00:00",
                    "strategy_id": "short_asset",
                    "multiplier": 0.5,
                    "wallet": 100.0,
                    "action_binds": 1.0,
                },
                {
                    "timestamp": "2026-05-01 01:00:00+00:00",
                    "strategy_id": "short_boll",
                    "multiplier": 0.5,
                    "wallet": 100.0,
                    "action_binds": 1.0,
                },
            ]
        ),
    ).set_index("strategy_id")

    assert bool(scored.loc["short_asset", "accepted"]) is False
    assert scored.loc["short_asset", "reject_reason"] == "action_value_safety_below_threshold"
    assert bool(scored.loc["short_boll", "accepted"]) is True
