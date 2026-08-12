from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.stage_i_meta_feature_challenger import (
    checkpoint_meta_feature_plan,
    evaluate_frozen_base_meta_feature_challenge,
    load_completed_stage_i_meta_selection,
    materialize_meta_feature_challenge,
)
from extreme_price_movements.stage_i_nested_feature_challenger import (
    MetaTargetMetricSpec,
    NestedFeatureChallengerError,
    StrictOOFResult,
)


def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    selector = tmp_path / "selector"
    selector.mkdir()
    (selector / "manifest.json").write_text(
        json.dumps({"schema": "selector", "status": "complete"})
    )
    (selector / "selector_feature_contract.json").write_text(
        json.dumps({"schema": "features"})
    )

    base_root = tmp_path / "base"
    base_side = base_root / "long"
    base_side.mkdir(parents=True)
    identity = pd.DataFrame(
        {
            "candidate_id": [f"c{i}" for i in range(30)],
            "__ts__": pd.date_range("2024-01-01", periods=30, freq="h", tz="UTC"),
            "__symbol__": ["BTC"] * 30,
        }
    )
    base = identity.copy()
    base["side_name"] = "long"
    base["decision_ts"] = base["__ts__"] + pd.Timedelta(hours=1)
    base["label_available_ts"] = base["decision_ts"] + pd.Timedelta(hours=12)
    base["exact_net_bps"] = np.linspace(-200, 200, len(base))
    base["r3_p_adverse"] = 0.2
    base["r3_p_weak"] = 0.3
    base["r3_p_clear"] = 0.5
    base["r3_opportunity_score"] = 0.3
    base_oof = base_side / "selector_base_oof.parquet"
    base.to_parquet(base_oof, index=False)
    base_manifest = base_side / "manifest.json"
    base_manifest.write_text(
        json.dumps(
            {
                "schema": "stage_i_base_feature_selection_v1",
                "status": "complete",
                "side": "long",
                "correlation_policy": "grouped-preserve",
            }
        )
    )

    meta = tmp_path / "meta" / "long"
    round_dir = meta / "mda" / "meta__long" / "side_long" / "round_01"
    round_dir.mkdir(parents=True)
    required = [
        "r3_p_adverse",
        "r3_p_weak",
        "r3_p_clear",
        "r3_opportunity_score",
        "base_r3_entropy",
        "base_r3_top2_margin",
        "base_r3_max_probability",
    ]
    optional = [f"context_family_{index:03d}" for index in range(73)]
    inputs = required + optional
    pd.DataFrame(
        {
            "feature": inputs,
            "mda_median": np.linspace(2.0, 0.1, len(inputs)),
            "mda_mean": np.linspace(2.0, 0.1, len(inputs)),
            "mda_positive_cohort_rate": 1.0,
            "mda_worst_cohort_mda": 0.1,
            "mda_latest_cohort_mda": 0.1,
            "mda_cohort_count": 3,
            "mda_n_repeats": 3,
            "confidence_label": "strong_keep",
        }
    ).to_csv(round_dir / "mda_feature_audit.csv", index=False)
    (round_dir / "mda_feature_selection_report.json").write_text(
        json.dumps(
            {"feature_audit_path": str(round_dir / "mda_feature_audit.csv")}
        )
    )
    candidate = identity.copy()
    candidate["side_name"] = "long"
    candidate["decision_ts"] = base.decision_ts
    candidate["selected_base_candidate"] = np.arange(len(candidate)) < 15
    candidate["base_candidate_fraction"] = 0.5
    candidate_path = meta / "base_candidate_handoff_audit.parquet"
    candidate.to_parquet(candidate_path, index=False)
    population = base.iloc[:15].loc[
        :,
        [
            "candidate_id",
            "__ts__",
            "__symbol__",
            "decision_ts",
            "side_name",
            "label_available_ts",
            "exact_net_bps",
        ],
    ].copy()
    population["target"] = (population.exact_net_bps > 0).astype(float)
    meta_oof = meta / "selector_meta_oof.parquet"
    population.to_parquet(meta_oof, index=False)
    map_path = meta / "prequential_value_map_audit.parquet"
    population.loc[:, ["candidate_id", "__ts__"]].to_parquet(map_path, index=False)
    manifest = {
        "schema": "stage_i_meta_feature_selection_v1",
        "status": "complete",
        "side": "long",
        "selected_features": required + optional[:18],
        "selected_feature_contract": required + optional[:18],
        "input_feature_contract": inputs,
        "required_same_side_base_oof_handoff_features": required,
        "selector_sample_manifest_sha256": _sha(selector / "manifest.json"),
        "selector_feature_contract_sha256": _sha(
            selector / "selector_feature_contract.json"
        ),
        "base_selector_manifest_sha256": _sha(base_manifest),
        "base_selector_oof_sha256": _sha(base_oof),
        "selector_meta_oof_sha256": _sha(meta_oof),
        "prequential_value_map_audit_sha256": _sha(map_path),
        "base_candidate_handoff_audit_sha256": _sha(candidate_path),
        "hpo_oof_score_semantics": "frozen_base_plus_target_specific_meta",
        "correlation_policy": "grouped-preserve",
        "base_correlation_policy": "grouped-preserve",
        "base_correlation_lineage": {"correlation_policy": "grouped-preserve"},
    }
    (meta / "manifest.json").write_text(json.dumps(manifest))
    return selector, base_root, meta


def test_meta_ladder_preserves_base_trust_and_is_nested_and_promotion_eligible(
    tmp_path: Path,
) -> None:
    selector, base, meta = _fixture(tmp_path)
    source = load_completed_stage_i_meta_selection(
        meta, side="long", selector_dir=selector, base_selection_dir=base
    )
    plan = materialize_meta_feature_challenge(source)
    sets = {item.name: item for item in plan.feature_sets}
    assert sets["automatic_sparse"].features == source.selected_features
    assert [len(sets[name].features) for name in ("top20", "top30", "top40", "top60")] == [20, 30, 40, 60]
    assert sets["top30"].features[:20] == sets["top20"].features
    assert sets["top40"].features[:30] == sets["top30"].features
    assert sets["top60"].features[:40] == sets["top40"].features
    assert set(source.required_base_trust_features).issubset(sets["top20"].features)
    assert sets["full_input_control"].features == source.input_features
    assert sets["full_input_control"].promotion_eligible is True
    assert plan.frozen_base_oof_sha256 == _sha(base / "long" / "selector_base_oof.parquet")
    checkpoint = checkpoint_meta_feature_plan(plan, tmp_path / "checkpoint")
    assert checkpoint_meta_feature_plan(plan, checkpoint) == checkpoint


def test_meta_loader_rejects_hash_drift_and_missing_full_input_contract(
    tmp_path: Path,
) -> None:
    selector, base, meta = _fixture(tmp_path)
    manifest_path = meta / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("input_feature_contract")
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(NestedFeatureChallengerError, match="contracts are incomplete"):
        load_completed_stage_i_meta_selection(
            meta, side="long", selector_dir=selector, base_selection_dir=base
        )


def test_meta_loader_rejects_explicit_base_correlation_policy_mismatch(
    tmp_path: Path,
) -> None:
    selector, base, meta = _fixture(tmp_path)
    manifest_path = meta / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["correlation_policy"] = "pre-mda-spearman-representative"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(NestedFeatureChallengerError, match="correlation-policy"):
        load_completed_stage_i_meta_selection(
            meta, side="long", selector_dir=selector, base_selection_dir=base
        )


def test_frozen_base_evaluator_holds_population_constant_across_meta_counts(
    tmp_path: Path,
) -> None:
    selector, base, meta = _fixture(tmp_path)
    source = load_completed_stage_i_meta_selection(
        meta, side="long", selector_dir=selector, base_selection_dir=base
    )
    plan = materialize_meta_feature_challenge(source)
    observed: list[tuple[str, tuple[str, ...]]] = []

    def hook(feature_set, frozen, spec):
        observed.append((feature_set.name, tuple(frozen.candidate_id)))
        frame = frozen.loc[:, ["candidate_id", "__ts__", "__symbol__", "exact_net_bps"]].copy()
        frame[spec.target_column] = (frame.exact_net_bps > 0).astype(float)
        frame[spec.prediction_columns[0]] = np.linspace(0.1, 0.9, len(frame))
        return StrictOOFResult(
            frame, {"strict_oof": True, "side": "long", "layer": "meta"}
        )

    result = evaluate_frozen_base_meta_feature_challenge(
        plan,
        meta_selection_dir=meta,
        frozen_base_oof_path=base / "long" / "selector_base_oof.parquet",
        meta_hook=hook,
        meta_specs=(
            MetaTargetMetricSpec(
                "reliability", "reliability", "y", ("p",)
            ),
        ),
    )
    assert result["comparison_scope"] == "frozen_base_sequential_meta_feature_count_only"
    assert len(result["evaluations"]) == 6
    assert len({rows for _, rows in observed}) == 1
    assert [name for name, _ in observed] == [item.name for item in plan.feature_sets]
