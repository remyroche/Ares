from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.packb_side_local_fs_hpo_stage import (
    HPOTrialEvaluation,
)
from extreme_price_movements.packb_static_point_feature_loader import (
    LoaderEvidenceBundle,
)
from scripts import run_packb_pre_march_side_fs_hpo as runner


def _labels() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c"],
            "side_name": ["long", "long", "long"],
            "__symbol__": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
            "__ts__": pd.to_datetime(
                [
                    "2025-11-01T00:00:00Z",
                    "2025-11-01T01:00:00Z",
                    "2025-11-01T02:00:00Z",
                ]
            ),
            runner.TARGET_COLUMN: [0.1, 0.9, 0.4],
            runner.WEIGHT_COLUMN: [0.5, 1.0, 0.8],
            runner.ECONOMIC_COLUMN: [-0.01, 0.03, 0.005],
        }
    )


def test_exact_label_loader_preserves_order_and_all_exact_keys(tmp_path: Path) -> None:
    labels = _labels()
    path = tmp_path / "labels.parquet"
    labels.to_parquet(path, index=False)
    ledger = labels.loc[
        [2, 0],
        ["candidate_id", "side_name", "__symbol__", "__ts__"],
    ].reset_index(drop=True)
    loader = runner.ExactLabelLoader([path])

    loaded = loader.load(ledger)

    assert loaded[runner.TARGET_COLUMN].tolist() == [0.4, 0.1]
    assert loader.weights(ledger, loaded[runner.TARGET_COLUMN]).tolist() == [
        0.8,
        0.5,
    ]
    assert loader.economic(ledger).tolist() == [0.005, -0.01]


def test_exact_label_loader_rejects_candidate_identity_disagreement(
    tmp_path: Path,
) -> None:
    labels = _labels()
    path = tmp_path / "labels.parquet"
    labels.to_parquet(path, index=False)
    ledger = labels.loc[:, ["candidate_id", "side_name", "__symbol__", "__ts__"]].copy()
    ledger.loc[0, "__symbol__"] = "WRONG"

    with pytest.raises(
        runner.PackBSideFSHPORunnerError,
        match="disagrees on side, symbol, or signal timestamp",
    ):
        runner.ExactLabelLoader([path]).load(ledger)


def test_explicit_hpo_design_is_side_local_deterministic_and_unique() -> None:
    long_a = runner.make_hpo_trials(side="long")
    long_b = runner.make_hpo_trials(side="long")
    short = runner.make_hpo_trials(side="short")

    assert len(long_a) == 150
    assert [trial.params for trial in long_a] == [trial.params for trial in long_b]
    assert (
        len(
            {
                json.dumps(trial.params, sort_keys=True, separators=(",", ":"))
                for trial in long_a
            }
        )
        == 150
    )
    assert [trial.params for trial in long_a] != [trial.params for trial in short]


def test_economic_objective_rewards_correct_ranking_and_net_lift() -> None:
    target = np.asarray([0.0, 0.1, 0.8, 1.0], dtype=np.float64)
    weights = np.ones(4, dtype=np.float64)
    net = np.asarray([-0.02, -0.01, 0.02, 0.04], dtype=np.float64)

    good = runner._economic_objective(target, target, weights, net)
    bad = runner._economic_objective(target[::-1], target, weights, net)

    assert good["objective"] > bad["objective"]
    assert good["weighted_rank_ic"] == pytest.approx(1.0)
    assert good["top10_mean_net_return"] == pytest.approx(0.04)


def test_hpo_selector_requires_three_folds_and_uses_worst_fold_tie_break() -> None:
    trials = runner.make_hpo_trials(side="long", count=2)
    evaluations = []
    for fold, objective_a, objective_b in (
        ("hpo_1", 0.3, 0.2),
        ("hpo_2", 0.1, 0.2),
        ("hpo_3", 0.2, 0.2),
    ):
        evaluations.extend(
            [
                HPOTrialEvaluation(
                    trial_id=trials[0].trial_id,
                    params=trials[0].params,
                    fold_name=fold,
                    result={"objective": objective_a},
                ),
                HPOTrialEvaluation(
                    trial_id=trials[1].trial_id,
                    params=trials[1].params,
                    fold_name=fold,
                    result={"objective": objective_b},
                ),
            ]
        )

    selected = runner.SideHPOSelector(side="long", trials=trials)(evaluations)

    assert selected["selected_trial_id"] == trials[1].trial_id
    assert selected["evaluated_trial_count"] == 2
    assert selected["evaluated_fold_count_per_trial"] == 3


def test_feature_provenance_binds_feature_and_loader_contract() -> None:
    contract = {
        "feature_columns": ["observable_a", "observable_b"],
        "generator_registry_sha256": "a" * 64,
        "raw_allowlist_sha256": "b" * 64,
        "selection_provenance": "causal",
        "source_schema_sha256": "c" * 64,
        "store_scan_manifest_sha256": "d" * 64,
        "feature_contract_sha256": "e" * 64,
    }
    bundle = LoaderEvidenceBundle(
        raw_universe_sha256="f" * 64,
        coverage_profile_sha256="0" * 64,
        feature_contract_sha256="e" * 64,
        loader_contract_sha256="1" * 64,
        loader_module_sha256="2" * 64,
        source_schema_sha256="c" * 64,
        source_revision="3" * 40,
    )

    provenance = runner._feature_provenance(contract, bundle)

    assert set(provenance) == {"observable_a", "observable_b"}
    assert (
        provenance["observable_a"]["causal_definition_sha256"]
        != provenance["observable_b"]["causal_definition_sha256"]
    )
    assert all(
        len(value) == 64 for entry in provenance.values() for value in entry.values()
    )


def test_canonical_label_files_require_bound_audit_and_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    labels = _labels()
    shard = labels_dir / "train_global_long_5_2025_11.parquet"
    labels.to_parquet(shard, index=False)
    audit_path = tmp_path / "audit.json"
    audit = {
        "status": "PASS",
        "failures": {},
        "per_file": [{"file": shard.name, "rows": len(labels)}],
    }
    audit_path.write_text(json.dumps(audit, sort_keys=True) + "\n", encoding="utf-8")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    manifest = {
        "input": {
            "canonical_shards": [shard.name],
            "causal_audit_path": str(audit_path),
            "causal_audit_sha256": runner.stage_manifest.sha256_file(audit_path),
        },
        "population_preflight": {"label_inventory": {"explicitly_excluded_shards": []}},
    }

    assert runner._canonical_label_files(labels_dir, manifest) == (shard,)

    changed = dict(audit)
    changed["per_file"] = [{"file": shard.name, "rows": len(labels) + 1}]
    audit_path.write_text(json.dumps(changed, sort_keys=True) + "\n", encoding="utf-8")
    manifest["input"]["causal_audit_sha256"] = runner.stage_manifest.sha256_file(
        audit_path
    )
    with pytest.raises(
        runner.PackBSideFSHPORunnerError, match="row count changed since audit"
    ):
        runner._canonical_label_files(labels_dir, manifest)
