from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.stage_i_nested_feature_challenger import (
    MetaTargetMetricSpec,
    NestedFeatureChallengerError,
    StrictOOFResult,
    checkpoint_nested_feature_plan,
    evaluate_nested_feature_challenge,
    load_completed_stage_i_base_selection,
    materialize_nested_feature_challenge,
)


def _source(tmp_path: Path, *, stable: bool = True, consistently_negative: bool = False) -> Path:
    root = tmp_path / "long"
    audit_dir = root / "mda" / "base__long__R3" / "side_long" / "round_01"
    audit_dir.mkdir(parents=True)
    features = ["alpha_one", "alpha_two", "beta_one", "gamma_one", "delta_one", "epsilon_one"]
    features.extend(f"family_{index:02d}_signal" for index in range(98))
    pd.DataFrame({
        "feature": features,
        "mda_median": ([-1.0] * len(features) if consistently_negative else list(range(len(features), 0, -1))),
        "mda_mean": ([-1.0] * len(features) if consistently_negative else list(range(len(features), 0, -1))),
        "mda_positive_cohort_rate": [1.0 if stable else 0.0] * len(features),
        "mda_worst_cohort_mda": [1.0 if stable else -1.0] * len(features),
        "mda_latest_cohort_mda": ([-1.0] * len(features) if consistently_negative else [1] * len(features)),
        "mda_cohort_count": [3] * len(features), "mda_n_repeats": [3] * len(features),
        "confidence_label": ["strong_keep"] * len(features),
    }).to_csv(audit_dir / "mda_feature_audit.csv", index=False)
    (audit_dir / "mda_feature_selection_report.json").write_text(json.dumps({
        "feature_audit_path": str(audit_dir / "mda_feature_audit.csv"),
    }))
    (root / "manifest.json").write_text(json.dumps({
        "schema": "stage_i_base_feature_selection_v1", "status": "complete", "side": "long",
        "selected_feature_contract": ["alpha_one", "alpha_two"], "input_feature_contract": features,
    }))
    return root


def test_materializer_preserves_automatic_sparse_and_builds_diverse_nested_sets(tmp_path: Path) -> None:
    source = load_completed_stage_i_base_selection(_source(tmp_path), side="long")
    plan = materialize_nested_feature_challenge(
        source, required_features=["beta_one"], protected_features=["gamma_one"],
    )
    sets = {item.name: item for item in plan.feature_sets}
    assert sets["automatic_sparse"].features == ("alpha_one", "alpha_two", "beta_one", "gamma_one")
    assert len(sets["top20"].features) == 20
    assert sets["top20"].features[:2] == ("beta_one", "gamma_one")
    assert sets["top20"].requested_feature_count == 20
    assert sets["automatic_sparse"].family_composition["alpha_one"] == 1
    assert all(item.source_ranks["alpha_one"] == 1 for item in plan.feature_sets)
    checkpoint = checkpoint_nested_feature_plan(plan, tmp_path / "checkpoint")
    assert (checkpoint / "nested_feature_sets.json").is_file()
    assert checkpoint_nested_feature_plan(plan, checkpoint) == checkpoint


def test_fixed_count_ladder_is_independent_of_an_oversized_automatic_prefix(tmp_path: Path) -> None:
    root = _source(tmp_path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["selected_feature_contract"] = manifest["input_feature_contract"][:87]
    manifest_path.write_text(json.dumps(manifest))
    source = load_completed_stage_i_base_selection(root, side="long")
    plan = materialize_nested_feature_challenge(
        source, required_features=("beta_one",), protected_features=("gamma_one",),
    )
    sets = {item.name: item for item in plan.feature_sets}
    assert len(sets["automatic_sparse"].features) == 87
    assert [len(sets[name].features) for name in ("top20", "top30", "top40", "top60")] == [20, 30, 40, 60]
    assert sets["top30"].features[:20] == sets["top20"].features
    assert sets["top40"].features[:30] == sets["top30"].features
    assert sets["top60"].features[:40] == sets["top40"].features
    assert sets["top20"].features != sets["automatic_sparse"].features


def test_full_input_control_preserves_authorized_contract_and_is_promotion_eligible(tmp_path: Path) -> None:
    source = load_completed_stage_i_base_selection(_source(tmp_path), side="long")
    plan = materialize_nested_feature_challenge(source)
    control = next(item for item in plan.feature_sets if item.name == "full_input_control")
    assert control.features == source.input_features
    assert control.control_provenance["kind"] == "full_input_control"
    assert control.control_provenance["postscreen_bypass"] is True
    assert control.promotion_eligible is True
    assert control.control_provenance["promotion_policy"].startswith(
        "eligible_only_if_best_under_identical_strict_OOF_and_OOS_gates"
    )
    assert all(key.startswith("full_input_control__") for key in control.tier_composition)


def test_round_consolidation_uses_earlier_evaluated_evidence_not_untested(tmp_path: Path) -> None:
    root = _source(tmp_path)
    round1 = root / "mda" / "base__long__R3" / "side_long" / "round_01"
    round2 = round1.parent / "round_02"
    round2.mkdir()
    feature = "family_00_signal"
    audit = pd.read_csv(round1 / "mda_feature_audit.csv")
    audit.loc[audit.feature.eq(feature), "mda_median"] = 9.0
    audit.to_csv(round1 / "mda_feature_audit.csv", index=False)
    audit.loc[~audit.feature.eq(feature)].to_csv(round2 / "mda_feature_audit.csv", index=False)
    (round2 / "mda_feature_selection_report.json").write_text(json.dumps({
        "feature_audit_path": str(round2 / "mda_feature_audit.csv"),
    }))
    source = load_completed_stage_i_base_selection(root, side="long")
    rank = source.source_ranks[feature]
    assert rank.audit_observed is True
    assert rank.source_round == "round_01"
    assert rank.source_audit_path.endswith("round_01/mda_feature_audit.csv")
    plan = materialize_nested_feature_challenge(source)
    full = next(item for item in plan.feature_sets if item.name == "full_input_control")
    assert full.source_rank_evidence[feature]["source_round"] == "round_01"


def test_materializer_keeps_uncertain_tiers_but_excludes_consistently_negative_fields(tmp_path: Path) -> None:
    source = load_completed_stage_i_base_selection(_source(tmp_path, stable=False), side="long")
    plan = materialize_nested_feature_challenge(source)
    top20 = next(item for item in plan.feature_sets if item.name == "top20")
    assert top20.tier_composition["borderline_or_uncertain"] > 0
    negative = load_completed_stage_i_base_selection(
        _source(tmp_path / "negative", stable=False, consistently_negative=True), side="long",
    )
    with pytest.raises(NestedFeatureChallengerError, match="consistently materially negative"):
        materialize_nested_feature_challenge(negative)


def test_materializer_records_pre_mda_fields_as_untested_instead_of_dropping_them(tmp_path: Path) -> None:
    root = _source(tmp_path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["input_feature_contract"].append("pre_mda_only")
    manifest_path.write_text(json.dumps(manifest))
    source = load_completed_stage_i_base_selection(root, side="long")
    rank = source.source_ranks["pre_mda_only"]
    assert rank.tier == "untested_or_group_skipped"
    assert rank.audit_observed is False


def _base_result(feature_set) -> StrictOOFResult:
    n = 12
    probabilities = np.tile(np.asarray([0.2, 0.3, 0.5]), (n, 1))
    frame = pd.DataFrame({
        "candidate_id": np.arange(n), "__ts__": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
        "__symbol__": ["BTC"] * n, "r3_class": np.arange(n) % 3, "exact_net_bps": np.linspace(-5, 8, n),
        "r3_p_adverse": probabilities[:, 0], "r3_p_weak": probabilities[:, 1], "r3_p_clear": probabilities[:, 2],
        "base_score_bps": np.linspace(-1, 1, n),
    })
    return StrictOOFResult(frame, {"strict_oof": True, "side": "long", "layer": "base"})


def _meta_result(feature_set, base: StrictOOFResult, spec: MetaTargetMetricSpec) -> StrictOOFResult:
    frame = base.frame.loc[:, ["candidate_id", "__ts__", "__symbol__", "exact_net_bps"]].copy()
    if spec.family == "ordinal":
        frame[spec.target_column] = np.arange(len(frame)) % 3
        for index, column in enumerate(spec.prediction_columns):
            frame[column] = [0.2, 0.3, 0.5][index]
    elif spec.family == "clipped_residual":
        frame[spec.target_column] = np.linspace(-1, 1, len(frame))
        frame[spec.prediction_columns[0]] = frame[spec.target_column]
    else:
        frame[spec.target_column] = np.arange(len(frame)) % 2
        frame[spec.prediction_columns[0]] = 0.5
    return StrictOOFResult(frame, {"strict_oof": True, "side": "long", "layer": "meta"})


def test_evaluator_uses_r3_primary_and_target_specific_meta_metrics(tmp_path: Path) -> None:
    source = load_completed_stage_i_base_selection(_source(tmp_path), side="long")
    plan = materialize_nested_feature_challenge(source)
    result = evaluate_nested_feature_challenge(plan, base_hook=_base_result, meta_hook=_meta_result, meta_specs=(
        MetaTargetMetricSpec("reliable", "reliability", "y", ("p",)),
        MetaTargetMetricSpec("veto", "overestimate_veto", "y", ("p",)),
        MetaTargetMetricSpec("ordinal", "ordinal", "y", ("p0", "p1", "p2")),
        MetaTargetMetricSpec("residual", "clipped_residual", "y", ("pred",), clip_bounds=(-1, 1)),
    ))
    one = result["evaluations"][0]
    assert {"multiclass_log_loss", "multiclass_brier", "exact_net_top_10_bps"}.issubset(one["base"])
    assert "ece_10" in one["meta"]["reliable"]["metrics"]
    assert "veto_false_negative_rate" in one["meta"]["veto"]["metrics"]
    assert "ordinal_expected_mae" in one["meta"]["ordinal"]["metrics"]
    assert "clipped_residual_mae" in one["meta"]["residual"]["metrics"]
    assert "huber" not in json.dumps(one["meta"]).lower()


def test_evaluator_rejects_meta_rows_that_do_not_match_base(tmp_path: Path) -> None:
    source = load_completed_stage_i_base_selection(_source(tmp_path), side="long")
    plan = materialize_nested_feature_challenge(source)

    def bad_meta(feature_set, base, spec):
        output = _meta_result(feature_set, base, spec)
        return StrictOOFResult(output.frame.iloc[:-1], output.provenance)

    with pytest.raises(NestedFeatureChallengerError, match="identical strict OOF rows"):
        evaluate_nested_feature_challenge(
            plan, base_hook=_base_result, meta_hook=bad_meta,
            meta_specs=(MetaTargetMetricSpec("reliable", "reliability", "y", ("p",)),),
        )
