from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.stage_i_base_feature_ladder import (
    BaseFeatureLadderConfig,
    BaseFeatureLadderInput,
    StageIBaseFeatureLadderError,
    run_pooled_base_feature_ladder,
    run_side_base_feature_ladder,
)
from extreme_price_movements.stage_i_nested_feature_challenger import (
    NESTED_SET_NAMES,
    NestedFeatureChallengePlan,
    NestedFeatureSet,
)


def _plan(side: str) -> NestedFeatureChallengePlan:
    sets = []
    for index, name in enumerate(NESTED_SET_NAMES):
        features = ("x1",) if name in {"automatic_sparse", "top20"} else ("x1", "x2")
        sets.append(NestedFeatureSet(
            side=side, name=name, requested_feature_count=None, features=features,
            added_features=(), source_ranks={key: idx + 1 for idx, key in enumerate(features)},
            feature_families={key: key for key in features},
            family_composition={key: 1 for key in features},
            tier_composition={"test": len(features)}, source_hash=f"{side}-{name}",
            promotion_eligible=True,
        ))
    return NestedFeatureChallengePlan(
        side=side, source_manifest_sha256="a" * 64, source_audit_sha256="b" * 64,
        source_audit_path="audit", required_features=(), protected_features={},
        stability_policy={}, feature_sets=tuple(sets),
    )


def _input(side: str, *, n: int = 72) -> BaseFeatureLadderInput:
    ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    frame = pd.DataFrame({
        "candidate_id": [f"{side}-{index}" for index in range(n)],
        "__ts__": ts, "__symbol__": ["BTC"] * n, "side_name": [side] * n,
        "decision_ts": ts + pd.Timedelta(hours=1),
        "label_available_ts": ts + pd.Timedelta(hours=13),
        "r3_class": np.arange(n) % 3,
        "exact_net_bps": np.linspace(-200.0, 200.0, n),
        "x1": np.arange(n, dtype=float), "x2": np.arange(n, dtype=float) * 2.0,
    })
    return BaseFeatureLadderInput(side=side, frame=frame, base_feature_universe=("x1", "x2"))


def _predictor(train: pd.DataFrame, target: np.ndarray, valid: pd.DataFrame, feature_set: NestedFeatureSet) -> np.ndarray:
    # The output depends only on permitted training labels and validation-time
    # feature values.  It therefore makes held-label mutation assertions sharp.
    offset = float(np.mean(target)) / 20.0
    value = valid.x1.to_numpy(float)
    clear = np.clip(0.25 + offset + value / max(1.0, value.max()) * 0.20, 0.05, 0.75)
    adverse = np.full(len(valid), 0.20)
    weak = 1.0 - clear - adverse
    return np.column_stack([adverse, weak, clear])


def _run(tmp_path: Path, side: str, *, resume: bool = False, payload: BaseFeatureLadderInput | None = None):
    return run_side_base_feature_ladder(
        payload or _input(side), _plan(side), base_predictor=_predictor,
        source_base_params={"n_estimators": 12, "objective": "multiclass", "num_class": 3},
        source_base_manifest_sha256="m" * 64, output_dir=tmp_path / side,
        config=BaseFeatureLadderConfig(n_validation_folds=4, min_train_rows=12), resume=resume,
    )


def test_base_ladder_is_meta_free_and_all_counts_share_full_strict_oof_population(tmp_path: Path) -> None:
    manifest = _run(tmp_path, "long")
    assert manifest["base_only"] is True
    assert manifest["meta_dependency"] == "forbidden"
    paths = [tmp_path / "long" / "arms" / name / "base_oof_predictions.parquet" for name in NESTED_SET_NAMES]
    frames = [pd.read_parquet(path) for path in paths]
    reference = frames[0].loc[:, ["candidate_id", "__ts__", "__symbol__", "fold_id"]]
    assert all(frame.loc[:, ["candidate_id", "__ts__", "__symbol__", "fold_id"]].equals(reference) for frame in frames[1:])
    assert len(reference) > 0
    for name in NESTED_SET_NAMES:
        provenance = pd.read_parquet(tmp_path / "long" / "arms" / name / "fold_provenance.parquet")
        assert provenance.strict_prior_resolved.all()
        assert provenance.meta_training_or_candidate_gate_used.eq(False).all()
        assert (
            pd.to_datetime(provenance.train_max_label_available_utc, utc=True)
            < pd.to_datetime(provenance.validation_start_utc, utc=True)
        ).all()
        request = json.loads((tmp_path / "long" / "arms" / name / "count_specific_base_hpo_refit_request.json").read_text())
        assert request["freeze_blocker"] == "count_specific_base_HPO_and_refit_required"
        assert request["fixed_source_hpo_diagnostic"] is True
        assert request["full_input_promotion_eligible"] is True


def test_base_oof_is_simplex_direct_score_and_same_fold_is_immune_to_held_label_mutation(tmp_path: Path) -> None:
    _run(tmp_path, "long")
    first = pd.read_parquet(tmp_path / "long" / "arms" / "automatic_sparse" / "base_oof_predictions.parquet")
    original = _input("long")
    changed = original.frame.copy()
    first_fold_ids = set(first.loc[first.fold_id.eq(0), "candidate_id"])
    changed.loc[changed.candidate_id.isin(first_fold_ids), "r3_class"] = 2
    _run(tmp_path / "changed", "long", payload=BaseFeatureLadderInput("long", changed, ("x1", "x2")))
    second = pd.read_parquet(tmp_path / "changed" / "long" / "arms" / "automatic_sparse" / "base_oof_predictions.parquet")
    initial = first.loc[first.fold_id.eq(0)].reset_index(drop=True)
    mutated = second.loc[second.fold_id.eq(0)].reset_index(drop=True)
    assert initial.loc[:, ["candidate_id", "fold_id"]].equals(mutated.loc[:, ["candidate_id", "fold_id"]])
    assert np.allclose(initial.loc[:, ["r3_p_adverse", "r3_p_weak", "r3_p_clear"]], mutated.loc[:, ["r3_p_adverse", "r3_p_weak", "r3_p_clear"]])
    p = first.loc[:, ["r3_p_adverse", "r3_p_weak", "r3_p_clear"]].to_numpy()
    assert np.allclose(p.sum(axis=1), 1.0)
    assert np.allclose(first.r3_opportunity_score, first.r3_p_clear - first.r3_p_adverse)


def test_resume_checksum_drift_fails_closed(tmp_path: Path) -> None:
    _run(tmp_path, "long")
    assert _run(tmp_path, "long", resume=True)["restart_status"] == "reused_verified_complete"
    path = tmp_path / "long" / "side_raw_metrics.parquet"
    path.write_bytes(b"drift")
    with pytest.raises(StageIBaseFeatureLadderError, match="checksum drift"):
        _run(tmp_path, "long", resume=True)


def test_resume_rejects_changed_label_or_feature_values_even_when_identities_match(tmp_path: Path) -> None:
    original = _input("long")
    _run(tmp_path, "long", payload=original)
    changed = original.frame.copy()
    changed.loc[0, "r3_class"] = int((changed.loc[0, "r3_class"] + 1) % 3)
    with pytest.raises(StageIBaseFeatureLadderError, match="request/hash drift"):
        _run(
            tmp_path, "long", resume=True,
            payload=BaseFeatureLadderInput("long", changed, ("x1", "x2")),
        )
    changed_feature = original.frame.copy()
    changed_feature.loc[0, "x1"] += 0.5
    with pytest.raises(StageIBaseFeatureLadderError, match="request/hash drift"):
        _run(
            tmp_path, "long", resume=True,
            payload=BaseFeatureLadderInput("long", changed_feature, ("x1", "x2")),
        )


def test_resume_rejects_extra_or_partial_arm_inventory_and_persists_denominators(tmp_path: Path) -> None:
    manifest = _run(tmp_path, "long")
    assert set(manifest["planned_arm_inventory"]) == set(NESTED_SET_NAMES)
    denominator = pd.read_parquet(tmp_path / "long" / "denominator_audit.parquet")
    assert denominator.full_candidate_rows.iloc[0] == 72
    assert denominator.base_burn_in_unscored_rows.iloc[0] > 0
    (tmp_path / "long" / "arms" / "automatic_sparse" / "stale.txt").write_text("x")
    with pytest.raises(StageIBaseFeatureLadderError, match="stale, extra, or partial"):
        _run(tmp_path, "long", resume=True)


def test_pooled_base_ladder_maps_by_side_then_selects_once_globally(monkeypatch, tmp_path: Path) -> None:
    import extreme_price_movements.stage_i_base_feature_ladder as ladder

    _run(tmp_path, "long")
    _run(tmp_path, "short")

    def mapped(frame: pd.DataFrame, **_kwargs):
        output = frame.copy()
        output["causal_21d_side_expected_net_bps"] = np.where(output.side_name.eq("short"), 100.0, 10.0)
        output["causal_21d_side_admitted_ge_50bps"] = True
        return output, pd.DataFrame({"strictly_prior_resolved": [True]})

    monkeypatch.setattr(ladder, "apply_causal_21d_side_admission", mapped)
    result = run_pooled_base_feature_ladder(
        long_dir=tmp_path / "long", short_dir=tmp_path / "short", output_dir=tmp_path / "pooled",
    )
    assert result["base_only"] is True
    metrics = pd.read_parquet(tmp_path / "pooled" / "pooled_global_metrics.parquet")
    admitted = metrics.loc[
        metrics.comparison.eq("with_admission_mapped_pooled_global") & metrics.top_fraction_of_original_population.eq(0.01)
    ]
    # Requested K is 1% of the 144 full side-candidate rows.  Only the
    # strictly-valid OOF rows can be actioned, but burn-in does not silently
    # shrink the requested global top-k denominator.
    assert admitted.full_candidate_population_rows.eq(144).all()
    assert admitted.strict_oof_scored_population_rows.eq(96).all()
    assert admitted.requested_rows_from_full_candidate_denominator.eq(2).all()
    assert admitted.selected_short_rows.eq(2).all()
    assert admitted.selected_long_rows.eq(0).all()


def test_cli_declares_no_meta_arguments_or_runtime_dependency() -> None:
    source = (Path(__file__).parents[1] / "scripts" / "run_stage_i_base_feature_ladder.py").read_text()
    assert "--meta-params" not in source and "--meta-arms" not in source
    assert "meta_params" not in source
    assert "fixed_lgbm_meta_predictor" not in source
