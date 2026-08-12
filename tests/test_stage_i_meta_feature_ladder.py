from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.stage_i_meta_feature_challenger import MetaFeatureChallengePlan
from extreme_price_movements.stage_i_meta_feature_ladder import (
    StageIMetaFeatureLadderError,
    _hpo_refit_request,
    _side_resume_verified,
    prepare_full_meta_ladder_population,
    run_pooled_meta_feature_ladder,
    run_strict_candidate_meta_feature_arm,
)
from extreme_price_movements.stage_i_meta_target_funnel import MetaTargetSpec
from extreme_price_movements.stage_i_nested_feature_challenger import NestedFeatureSet


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _frame() -> tuple[pd.DataFrame, np.ndarray]:
    n = 18
    ts = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    frame = pd.DataFrame({
        "candidate_id": [f"c{i}" for i in range(n)],
        "__ts__": ts,
        "__symbol__": ["BTC"] * n,
        "candidate_key": [f"long::c{i}" for i in range(n)],
        "side_name": ["long"] * n,
        "decision_ts": ts + pd.Timedelta(hours=1),
        "label_available_ts": ts + pd.Timedelta(hours=13),
        "exact_net_bps": np.linspace(-180, 180, n),
        "r3_opportunity_score": np.linspace(-0.2, 0.4, n),
        "prequential_base_expected_net_bps": np.linspace(-50, 50, n),
        "candidate_selected": np.arange(n) % 2 == 0,
        "valid_resolved_target": True,
        "mapping_reference_eligible": True,
        "original_side_population_rows": n + 7,
        "x": np.linspace(-1, 1, n),
    })
    fold = np.repeat(np.array([-1, 0, 1], dtype=np.int32), 6)
    return frame, fold


def _tercile_predictor(train, target, weight, valid, spec):
    # Deliberately ignores all validation outcomes.  It is sufficient for the
    # target-lineage test and still returns a proper classifier simplex.
    p = np.tile([0.2, 0.6, 0.2], (len(valid), 1))
    return p


def test_strict_candidate_arm_scores_full_population_but_fits_only_prior_candidates() -> None:
    frame, fold = _frame()
    prediction, provenance = run_strict_candidate_meta_feature_arm(
        frame,
        MetaTargetSpec("t3q", "quantile_ordinal_residual", residual_clip_bps=200, shrinkage_support=5),
        feature_columns=("x",),
        fold_id=fold,
        predictor=_tercile_predictor,
        min_train_candidate_rows=3,
    )
    # 12 held-out full base rows are scored, whereas only the pre-frozen
    # candidate rows supplied target fitting.
    assert len(prediction) == 12
    assert provenance.validation_scored_full_rows.tolist() == [6, 6]
    assert provenance.train_candidate_rows.tolist() == [3, 6]
    assert prediction.action_admitted.sum() == 6
    assert prediction.candidate_selected.sum() == 6
    assert provenance.strict_prior_resolved.all()
    assert provenance.mapping_reference_scope.eq(
        "complete_side_base_oof_population_not_top30_prefilter"
    ).all()


def test_held_out_target_mutation_cannot_change_same_fold_t3q_prediction() -> None:
    frame, fold = _frame()
    spec = MetaTargetSpec("t3q", "quantile_ordinal_residual", residual_clip_bps=200, shrinkage_support=5)
    first, first_provenance = run_strict_candidate_meta_feature_arm(
        frame, spec, feature_columns=("x",), fold_id=fold,
        predictor=_tercile_predictor, min_train_candidate_rows=3,
    )
    mutated = frame.copy()
    # Fold 0 is held out at its own prediction time.  Its label may affect a
    # later fold, but cannot alter fold-0 q33/q67, prediction or score.
    mutated.loc[fold == 0, "exact_net_bps"] *= -50.0
    second, second_provenance = run_strict_candidate_meta_feature_arm(
        mutated, spec, feature_columns=("x",), fold_id=fold,
        predictor=_tercile_predictor, min_train_candidate_rows=3,
    )
    first_fold = first.fold_id.eq(0)
    second_fold = second.fold_id.eq(0)
    assert np.allclose(
        first.loc[first_fold, "score"], second.loc[second_fold, "score"]
    )
    assert np.allclose(
        first_provenance.loc[first_provenance.fold_id.eq(0), ["residual_q33_bps", "residual_q67_bps"]],
        second_provenance.loc[second_provenance.fold_id.eq(0), ["residual_q33_bps", "residual_q67_bps"]],
    )


def test_full_population_builder_does_not_conflate_top30_with_mapping_support(monkeypatch) -> None:
    import extreme_price_movements.stage_i_meta_feature_ladder as ladder

    n = 8
    ts = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    ledger = pd.DataFrame({
        "candidate_id": [f"c{i}" for i in range(n)], "__ts__": ts,
        "__symbol__": ["BTC"] * n, "side_name": ["long"] * n,
        "label_available_ts": ts + pd.Timedelta(hours=13),
        "exact_net_bps": np.arange(n, dtype=float),
    })
    raw = ledger.loc[:, ["candidate_id", "__ts__", "__symbol__"]].copy()
    raw["x"] = np.arange(n, dtype=float)
    base = ledger.loc[:, ["candidate_id", "__ts__", "__symbol__", "side_name", "label_available_ts", "exact_net_bps"]].copy()
    base["r3_p_adverse"], base["r3_p_weak"], base["r3_p_clear"] = 0.2, 0.3, 0.5
    handoff = ledger.loc[:, ["candidate_id", "__ts__", "__symbol__"]].copy()
    handoff["side_name"] = "long"
    handoff["selected_base_candidate"] = [True, True, False, False, False, False, False, False]
    monkeypatch.setattr(
        ladder, "prequential_same_side_r3_value_map",
        lambda **kwargs: (np.arange(len(kwargs["score"]), dtype=float), pd.DataFrame({"support": [1] * len(kwargs["score"])}), {}),
    )
    monkeypatch.setattr(
        ladder, "base_oof_trust_features",
        lambda probability, audit: pd.DataFrame({"trust": np.ones(len(probability))}),
    )
    model, _, audited = prepare_full_meta_ladder_population(
        ledger, raw, base, handoff, side="long"
    )
    assert len(model) == n
    assert model.mapping_reference_eligible.sum() == n
    assert model.candidate_selected.sum() == 2
    assert audited.mapping_reference_population.all()


def test_full_population_builder_rejects_candidate_identity_drift(monkeypatch) -> None:
    """A same-length handoff still cannot be paired positionally by accident."""
    import extreme_price_movements.stage_i_meta_feature_ladder as ladder

    n = 3
    ts = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    ledger = pd.DataFrame({
        "candidate_id": [f"c{i}" for i in range(n)], "__ts__": ts,
        "__symbol__": ["BTC"] * n, "side_name": ["long"] * n,
        "label_available_ts": ts + pd.Timedelta(hours=13),
        "exact_net_bps": np.arange(n, dtype=float),
    })
    raw = ledger.loc[:, ["candidate_id", "__ts__", "__symbol__"]].copy()
    base = ledger.loc[:, ["candidate_id", "__ts__", "__symbol__", "side_name", "label_available_ts", "exact_net_bps"]].copy()
    base["r3_p_adverse"], base["r3_p_weak"], base["r3_p_clear"] = 0.2, 0.3, 0.5
    handoff = ledger.loc[:, ["candidate_id", "__ts__", "__symbol__", "side_name"]].iloc[::-1].reset_index(drop=True)
    handoff["selected_base_candidate"] = True
    with pytest.raises(StageIMetaFeatureLadderError, match="identity order"):
        prepare_full_meta_ladder_population(ledger, raw, base, handoff, side="long")


def _feature_set(name: str, features: tuple[str, ...], *, eligible: bool = True) -> NestedFeatureSet:
    return NestedFeatureSet(
        "long", name, None, features, (), {name: 1},
        {feature: feature for feature in features}, {"f": len(features)},
        {"t": len(features)}, f"{name}-hash", promotion_eligible=eligible,
    )


def _plan() -> MetaFeatureChallengePlan:
    values = (
        _feature_set("automatic_sparse", ("x",)),
        _feature_set("full_input_control", ("x", "y"), eligible=True),
        _feature_set("top20", ("x",)), _feature_set("top30", ("x",)),
        _feature_set("top40", ("x",)), _feature_set("top60", ("x",)),
    )
    return MetaFeatureChallengePlan(
        side="long", source_manifest_sha256="a" * 64, source_audit_sha256="b" * 64,
        source_audit_path="audit", selector_manifest_sha256="c" * 64,
        selector_feature_contract_sha256="d" * 64, frozen_base_manifest_sha256="e" * 64,
        frozen_base_oof_sha256="f" * 64, candidate_handoff_audit_sha256="1" * 64,
        selector_meta_oof_sha256="2" * 64, target_semantics="test",
        required_base_trust_features=(), required_features=(), protected_features=(),
        feature_sets=values,
    )


def test_full_input_remains_promotion_eligible_but_count_hpo_blocks_freezing() -> None:
    plan = _plan()
    full = next(item for item in plan.feature_sets if item.name == "full_input_control")
    request = _hpo_refit_request(
        plan=plan, feature_set=full,
        spec=MetaTargetSpec("t3q", "quantile_ordinal_residual"),
        side="long", source_meta_params={"n_estimators": 20},
        model_features=("x",), ladder_request_sha256="a" * 64,
    )
    assert request["full_input_promotion_eligible"] is True
    assert request["freeze_eligible_now"] is False
    assert request["freeze_blocker"] == "count_specific_target_HPO_and_refit_required"


def test_resume_checksum_drift_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "arm"
    root.mkdir()
    for name in ("evaluation_population.parquet", "full_mapping_audit.parquet"):
        (root / name).write_bytes(name.encode())
    (root / "hpo_refit_requests.json").write_text("[]")
    manifest = {
        "schema": "stage_i_meta_feature_ladder_execution_v1", "status": "complete", "request_sha256": "request",
        "evaluation_population_sha256": _sha(root / "evaluation_population.parquet"),
        "full_mapping_audit_sha256": _sha(root / "full_mapping_audit.parquet"),
        "hpo_refit_requests_sha256": _sha(root / "hpo_refit_requests.json"),
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    assert _side_resume_verified(root, "request") is not None
    (root / "full_mapping_audit.parquet").write_bytes(b"drift")
    with pytest.raises(StageIMetaFeatureLadderError, match="checksum drift"):
        _side_resume_verified(root, "request")


def _canonical_sha(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _write_pooled_arm(root: Path, *, side: str, score: float, request_sha: str) -> dict:
    arm = root / "arms" / "full_input_control" / "t3q"
    arm.mkdir(parents=True)
    ts = pd.Timestamp("2024-02-01", tz="UTC")
    frame = pd.DataFrame({
        "candidate_id": [side], "__ts__": [ts], "__symbol__": ["BTC"],
        "candidate_key": [f"{side}::{side}"], "side_name": [side],
        "decision_ts": [ts], "label_available_ts": [ts + pd.Timedelta(hours=12)],
        "exact_net_bps": [float(score)], "score": [float(score)],
        "action_admitted": [True], "mapping_reference_eligible": [True],
        "candidate_selected": [True], "valid_resolved_target": [True],
        "original_side_population_rows": [50], "target": [0.0],
    })
    path = arm / "oof_predictions.parquet"
    provenance = arm / "fold_provenance.parquet"
    hpo_path = arm / "count_specific_hpo_refit_request.json"
    frame.to_parquet(path, index=False)
    pd.DataFrame({"fold_id": [0]}).to_parquet(provenance, index=False)
    expected = {
        "feature_set": "full_input_control", "feature_set_sha256": "full-hash",
        "target_arm_id": "t3q", "target_family": "quantile_ordinal_residual",
        "classifier_features": ["x"],
    }
    hpo = {
        "feature_set": expected["feature_set"],
        "feature_set_sha256": expected["feature_set_sha256"],
        "target_arm_id": expected["target_arm_id"],
        "target_family": expected["target_family"],
        "model_features": ["x"], "model_features_sha256": _canonical_sha(["x"]),
        "ladder_request_sha256": request_sha,
    }
    hpo["request_sha256"] = _canonical_sha(hpo)
    hpo_path.write_text(json.dumps(hpo))
    (arm / "manifest.json").write_text(json.dumps({
        "schema": "stage_i_meta_feature_ladder_execution_v1", "status": "complete",
        "request_sha256": request_sha,
        "feature_set": expected["feature_set"],
        "feature_set_sha256": expected["feature_set_sha256"],
        "target_arm_id": expected["target_arm_id"], "target_family": expected["target_family"],
        "classifier_feature_contract_sha256": _canonical_sha(["x"]),
        "oof_predictions_sha256": _sha(path),
        "fold_provenance_sha256": _sha(provenance),
        "count_specific_hpo_refit_request_sha256": _sha(hpo_path),
    }))
    return expected


def test_pooled_ladder_ranks_globally_only_after_mapping(monkeypatch, tmp_path: Path) -> None:
    import extreme_price_movements.stage_i_meta_feature_ladder as ladder

    long, short, out = tmp_path / "long", tmp_path / "short", tmp_path / "out"
    for root in (long, short):
        root.mkdir()
    long_plan = _write_pooled_arm(long, side="long", score=10.0, request_sha="long-request")
    short_plan = _write_pooled_arm(short, side="short", score=100.0, request_sha="short-request")
    for root, request_sha, plan in ((long, "long-request", long_plan), (short, "short-request", short_plan)):
        arm_manifest = root / "arms" / "full_input_control" / "t3q" / "manifest.json"
        arm = json.loads(arm_manifest.read_text())
        inventory = [{
            "feature_set": plan["feature_set"], "target_arm_id": plan["target_arm_id"],
            "arm_manifest_sha256": _sha(arm_manifest),
            "oof_predictions_sha256": arm["oof_predictions_sha256"],
            "fold_provenance_sha256": arm["fold_provenance_sha256"],
            "count_specific_hpo_refit_request_sha256": arm["count_specific_hpo_refit_request_sha256"],
        }]
        (root / "manifest.json").write_text(json.dumps({
            "schema": "stage_i_meta_feature_ladder_execution_v1", "status": "complete",
            "request_sha256": request_sha, "planned_arm_inventory": [plan],
            "arm_inventory": inventory,
        }))

    def mapped(frame, **kwargs):
        result = frame.copy()
        result["causal_21d_side_expected_net_bps"] = result.score
        result["causal_21d_side_admitted_ge_50bps"] = True
        return result, pd.DataFrame({"strictly_prior_resolved": [True]})

    monkeypatch.setattr(ladder, "apply_causal_21d_side_admission", mapped)
    run_pooled_meta_feature_ladder(long_dir=long, short_dir=short, output_dir=out)
    metrics = pd.read_parquet(out / "pooled_global_metrics.parquet")
    admitted = metrics.loc[
        metrics.comparison.eq("with_admission_mapped_pooled_global") & metrics.top_fraction.eq(0.01)
    ].iloc[0]
    assert admitted.selected_short_rows == 1
    assert admitted.selected_long_rows == 0
    assert admitted.full_input_promotion_eligible
    assert not admitted.freeze_eligible_now
