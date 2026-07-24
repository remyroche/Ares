from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.packb_side_local_fs_hpo_stage import (
    FeatureSelectionInput,
    HPOTrialEvaluation,
    StageDataset,
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
            "__decision_ts__": pd.to_datetime(
                [
                    "2025-11-01T01:00:00Z",
                    "2025-11-01T02:00:00Z",
                    "2025-11-01T03:00:00Z",
                ]
            ),
            runner.TARGET_COLUMN: [0.1, 0.9, 0.4],
            runner.WEIGHT_COLUMN: [0.5, 1.0, 0.8],
            runner.ECONOMIC_COLUMN: [-0.01, 0.03, 0.005],
            runner.ARCHETYPE_COLUMN: ["mixed", "breakout_impulse", "mixed"],
            runner.NET_POSITIVE_COLUMN: [0.0, 1.0, 1.0],
            runner.MAE_TO_SL_COLUMN: [0.8, 0.2, 0.4],
            runner.TIMEOUT_COLUMN: [0.0, 0.0, 1.0],
        }
    )


def test_exact_label_loader_preserves_order_and_all_exact_keys(tmp_path: Path) -> None:
    labels = _labels()
    labels.loc[2, runner.MAE_TO_SL_COLUMN] = np.nan
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
    assert loader.selection_context(ledger)[runner.MAE_TO_SL_COLUMN].tolist() == [
        0.0,
        0.8,
    ]


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


def test_exact_label_loader_handles_mixed_naive_and_utc_signal_timestamp_shards(
    tmp_path: Path,
) -> None:
    april = _labels().iloc[[0]].copy()
    july = _labels().iloc[[1]].copy()
    april["__ts__"] = april["__ts__"].dt.tz_localize(None)
    april_path = tmp_path / "april.parquet"
    july_path = tmp_path / "july.parquet"
    april.to_parquet(april_path, index=False)
    july.to_parquet(july_path, index=False)
    ledger = july.loc[
        :, ["candidate_id", "side_name", "__symbol__", "__ts__"]
    ].reset_index(drop=True)

    loaded = runner.ExactLabelLoader([april_path, july_path]).load(ledger)

    assert loaded[runner.TARGET_COLUMN].tolist() == [0.9]


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

    timestamps = pd.to_datetime(["2025-01-01T00:00:00Z"] * 4)
    symbols = ["A", "B", "C", "D"]
    good = runner._economic_objective(
        target,
        target,
        weights,
        net,
        timestamps=timestamps,
        symbols=symbols,
    )
    bad = runner._economic_objective(
        target[::-1],
        target,
        weights,
        net,
        timestamps=timestamps,
        symbols=symbols,
    )

    assert good["objective"] > bad["objective"]
    assert good["weighted_rank_ic"] == pytest.approx(1.0)
    assert good["top10_mean_net_return"] == pytest.approx(0.04)


def test_recent_winner_selector_is_side_local_and_preserves_process_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    columns = [f"feature_{index}" for index in range(10)]
    train_ledger = pd.DataFrame(
        {
            "candidate_id": ["t0", "t1", "t2"],
            "side_name": ["short"] * 3,
            "__symbol__": ["A", "B", "C"],
            "__ts__": pd.to_datetime(
                [
                    "2025-10-01T00:00:00Z",
                    "2025-10-01T01:00:00Z",
                    "2025-10-01T02:00:00Z",
                ]
            ),
        }
    )
    valid_ledger = pd.DataFrame(
        {
            "candidate_id": ["v0", "v1"],
            "side_name": ["short"] * 2,
            "__symbol__": ["A", "B"],
            "__ts__": pd.to_datetime(["2025-11-01T00:00:00Z", "2025-11-01T01:00:00Z"]),
        }
    )
    train_features = pd.DataFrame(
        np.arange(30, dtype=np.float32).reshape(3, 10), columns=columns
    )
    valid_features = pd.DataFrame(
        np.arange(20, dtype=np.float32).reshape(2, 10), columns=columns
    )

    class Labels:
        @staticmethod
        def selection_context(ledger: pd.DataFrame) -> pd.DataFrame:
            rows = len(ledger)
            return pd.DataFrame(
                {
                    runner.ARCHETYPE_COLUMN: ["mixed"] * rows,
                    runner.NET_POSITIVE_COLUMN: np.arange(rows) % 2,
                    runner.MAE_TO_SL_COLUMN: np.full(rows, 0.5),
                    runner.TIMEOUT_COLUMN: np.zeros(rows),
                    runner.ECONOMIC_COLUMN: np.linspace(-0.01, 0.02, rows),
                }
            )

    captured: dict[str, object] = {}

    def fake_train(features: pd.DataFrame, target: np.ndarray, **kwargs: object):
        captured["features"] = features.copy()
        captured["target"] = target.copy()
        captured.update(kwargs)
        return {
            "selected_feature_names": columns[:8],
            "feature_stats": pd.DataFrame(
                {
                    "feature": columns,
                    "feature_score": np.linspace(1.0, 0.1, len(columns)),
                    "hard_drop": [False] * 8 + [True] * 2,
                }
            ),
            "metrics": {"J_final": 0.42},
            "selection_history": [{"round": 1}],
        }

    import extreme_price_movements.lgbm_pipeline as pipeline

    monkeypatch.setattr(pipeline, "train_lgbm_stability_candidate", fake_train)
    original_burn_in_days = pipeline.LGBM_BASE_FORWARD_BURN_IN_DAYS
    original_short_history_fallback = pipeline.LGBM_FORWARD_ALLOW_SHORT_HISTORY_FALLBACK
    selector = runner.RecentWinnerSideFeatureSelector(
        side="short",
        labels=Labels(),  # type: ignore[arg-type]
        seed=7,
    )
    result = selector(
        FeatureSelectionInput(
            side="short",
            candidate_features=tuple(columns),
            train=StageDataset(
                ledger=train_ledger,
                features=train_features,
                target=pd.Series([0.1, 0.2, 0.3]),
                weights=pd.Series([1.0, 2.0, 3.0]),
            ),
            validation=StageDataset(
                ledger=valid_ledger,
                features=valid_features,
                target=pd.Series([0.4, 0.5]),
                weights=pd.Series([4.0, 5.0]),
            ),
        )
    )

    assert result["selected_features"] == columns[:8]
    assert result["selection_scope"] == "side_local"
    assert result["fallback_used"] is False
    assert "mda" in result["selection_methods"]
    assert len(captured["features"]) == 5  # type: ignore[arg-type]
    assert np.array_equal(captured["sample_weight"], np.ones(5, dtype=np.float32))
    assert captured["cfg"]["mda_config"]["correlation_pruning_floor_count"] == 300
    assert set(captured["label_context"]["side_name"]) == {"short"}
    assert (
        result["recent_winner_alignment"]["forward_validation"]["burn_in_days"] == 180.0
    )
    assert (
        result["recent_winner_alignment"]["forward_validation"][
            "short_history_fallback"
        ]
        is False
    )
    assert pipeline.LGBM_BASE_FORWARD_BURN_IN_DAYS == original_burn_in_days
    assert (
        pipeline.LGBM_FORWARD_ALLOW_SHORT_HISTORY_FALLBACK
        == original_short_history_fallback
    )


def test_top_fraction_is_per_timestamp_with_symbol_tie_break() -> None:
    predictions = np.ones(20, dtype=np.float64)
    timestamps = pd.to_datetime(
        ["2025-01-01T00:00:00Z"] * 10 + ["2025-01-01T01:00:00Z"] * 10
    )
    symbols = [
        "J",
        "I",
        "H",
        "G",
        "F",
        "E",
        "D",
        "C",
        "B",
        "A",
    ] * 2

    selected = runner._top_fraction_indices(
        predictions, timestamps=timestamps, symbols=symbols
    )

    assert selected.tolist() == [9, 19]


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


def test_representation_loader_materializes_frozen_side_outputs_only_when_needed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def raw_loader(_ledger: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        calls.append(columns)
        values = {
            "raw_a": np.asarray([1.0, 2.0], dtype=np.float32),
            "raw_b": np.asarray([3.0, 4.0], dtype=np.float32),
        }
        return pd.DataFrame({column: values[column] for column in columns})

    def transform(
        raw: pd.DataFrame, _state: dict[str, object], *, index: pd.Index
    ) -> pd.DataFrame:
        assert list(raw.columns) == ["raw_a", "raw_b"]
        return pd.DataFrame(
            {
                "dae_b16_00": [0.1, 0.2],
                "gmm_ood_score": [0.3, 0.4],
            },
            index=index,
        )

    monkeypatch.setattr(runner, "transform_ae_gmm_features", transform)
    loader = runner.SideRepresentationFeatureLoader(
        raw_loader=raw_loader,
        raw_features=["raw_a", "raw_b"],
        state={"enabled": True},
        generated_features=["dae_b16_00", "gmm_ood_score"],
    )
    ledger = pd.DataFrame({"candidate_id": ["a", "b"]})

    represented = loader(ledger, ["raw_b", "gmm_ood_score"])
    raw_only = loader(ledger, ["raw_a"])

    assert represented["raw_b"].tolist() == [3.0, 4.0]
    assert represented["gmm_ood_score"].tolist() == pytest.approx([0.3, 0.4])
    assert raw_only.to_dict("list") == {"raw_a": [1.0, 2.0]}
    assert calls == [["raw_a", "raw_b"], ["raw_a"]]


def test_representation_loader_leaves_incomplete_rows_for_joint_coverage_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raw_loader(_ledger: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        values = {
            "raw_a": np.asarray([1.0, np.nan, 3.0], dtype=np.float32),
            "raw_b": np.asarray([4.0, 5.0, 6.0], dtype=np.float32),
        }
        return pd.DataFrame({column: values[column] for column in columns})

    def transform(
        raw: pd.DataFrame, _state: dict[str, object], *, index: pd.Index
    ) -> pd.DataFrame:
        assert list(index) == [0, 2]
        assert np.isfinite(raw.to_numpy()).all()
        return pd.DataFrame({"dae_b16_00": [0.1, 0.3]}, index=index)

    monkeypatch.setattr(runner, "transform_ae_gmm_features", transform)
    loader = runner.SideRepresentationFeatureLoader(
        raw_loader=raw_loader,
        raw_features=["raw_a", "raw_b"],
        state={"enabled": True},
        generated_features=["dae_b16_00"],
    )

    represented = loader(
        pd.DataFrame({"candidate_id": ["a", "b", "c"]}),
        ["raw_a", "dae_b16_00"],
    )

    assert represented["raw_a"].tolist()[0::2] == [1.0, 3.0]
    assert np.isnan(represented.loc[1, "raw_a"])
    assert represented["dae_b16_00"].tolist()[0::2] == pytest.approx([0.1, 0.3])
    assert np.isnan(represented.loc[1, "dae_b16_00"])


def test_fs_hpo_raw_loader_reads_only_requested_raw_subset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = {
        "schema": "packb_static_point_feature_loader_v1",
        "feature_columns": ["raw_a", "raw_b", "raw_c"],
        "candidate_universe_sha256": "a" * 64,
        "source_schema_sha256": "b" * 64,
        "raw_allowlist_sha256": "c" * 64,
        "generator_registry_sha256": "d" * 64,
        "store_scan_manifest_sha256": "e" * 64,
        "coverage_profile_sha256": "f" * 64,
        "min_exact_key_coverage": 1.0,
        "min_non_null_feature_coverage": 0.99,
        "max_feature_columns": 256,
        "coverage_admission_rejections": [],
        "feature_contract_sha256": "0" * 64,
    }
    canonical_calls: list[tuple[str, ...]] = []
    subset_calls: list[tuple[str, ...]] = []

    def canonical_factory(**_kwargs: object):
        def canonical(
            _ledger: pd.DataFrame, columns: list[str] | tuple[str, ...]
        ) -> pd.DataFrame:
            canonical_calls.append(tuple(columns))
            return pd.DataFrame(
                {
                    "raw_a": [1.0],
                    "raw_b": [2.0],
                    "raw_c": [3.0],
                }
            ).loc[:, list(columns)]

        return canonical

    def subset_load(
        _ledger: pd.DataFrame,
        *,
        feature_contract: dict[str, object],
        **_kwargs: object,
    ) -> pd.DataFrame:
        columns = tuple(feature_contract["feature_columns"])
        subset_calls.append(columns)
        values = {"raw_a": [1.0], "raw_b": [2.0], "raw_c": [3.0]}
        return pd.DataFrame({column: values[column] for column in columns})

    monkeypatch.setattr(runner, "make_packb_static_feature_loader", canonical_factory)
    monkeypatch.setattr(runner, "load_point_in_time_features", subset_load)
    loader = runner.make_fs_hpo_raw_feature_loader(
        feature_store_dir=Path("/feature-store"),
        feature_contract=parent,
        evidence_bundle=LoaderEvidenceBundle(
            raw_universe_sha256="1" * 64,
            coverage_profile_sha256="2" * 64,
            feature_contract_sha256="0" * 64,
            loader_contract_sha256="3" * 64,
            loader_module_sha256="4" * 64,
            source_schema_sha256="b" * 64,
            source_revision="5" * 40,
        ),
        resource_guard=object(),
    )
    ledger = pd.DataFrame({"candidate_id": ["a"]})

    subset = loader(ledger, ["raw_c", "raw_a"])
    full = loader(ledger, ["raw_a", "raw_b", "raw_c"])

    assert subset.to_dict("list") == {"raw_c": [3.0], "raw_a": [1.0]}
    assert full.to_dict("list") == {
        "raw_a": [1.0],
        "raw_b": [2.0],
        "raw_c": [3.0],
    }
    assert subset_calls == [("raw_a", "raw_c")]
    assert canonical_calls == [("raw_a", "raw_b", "raw_c")]
    assert len(loader.fs_hpo_subset_loading_contract_sha256) == 64


def test_active_ae_contract_excludes_row_order_temporal_outputs() -> None:
    columns = runner._active_ae_gmm_columns({"gmm_n_components": 3})

    assert len(columns) == runner.AE_GMM_LATENT_DIM + 3 * 3 + 11
    assert "gmm_cluster_posterior_2" in columns
    assert "gmm_posterior_delta_1" not in columns
    assert "dae_reconstruction_error_delta_1" not in columns


def test_generated_feature_provenance_binds_learned_state() -> None:
    contract = {
        "feature_columns": ["raw_a"],
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
    state = {
        "cycle_state_hash": "4" * 64,
        "input_feature_order_hash": "5" * 64,
    }

    provenance = runner._feature_provenance(
        contract,
        bundle,
        state=state,
        generated_features=["dae_b16_00"],
    )

    assert set(provenance) == {"raw_a", "dae_b16_00"}
    assert all(len(value) == 64 for value in provenance["dae_b16_00"].values())


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
