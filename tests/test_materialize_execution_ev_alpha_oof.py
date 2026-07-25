from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "materialize_execution_ev_alpha_oof",
    ROOT / "scripts" / "materialize_execution_ev_alpha_oof.py",
)
assert SPEC and SPEC.loader
materializer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = materializer
SPEC.loader.exec_module(materializer)


def _lineage_hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _lineage_frames(paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = pd.read_parquet(paths["candidates"])
    oof = pd.read_parquet(paths["oof"])
    candidate_identity = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(candidates["__ts__"], utc=True),
            "__symbol__": candidates["__symbol__"].astype(str).str.strip(),
            "side_name": candidates["side_name"].astype(str).str.lower().str.strip(),
            "candidate_id": candidates["candidate_id"].astype(str).str.strip(),
        }
    )
    oof_lineage = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(oof["__ts__"], utc=True),
            "__symbol__": oof["__symbol__"].astype(str).str.strip(),
            "side_name": oof["side_name"].astype(str).str.lower().str.strip(),
            "candidate_id": oof["candidate_id"].astype(str).str.strip(),
            "oof_fold": oof["oof_fold"].astype(str).str.strip(),
            "validation_start": pd.to_datetime(oof["validation_start"], utc=True),
            "train_decision_cutoff": pd.to_datetime(
                oof["train_decision_cutoff"], utc=True
            ),
            "label_resolution_available_at": pd.to_datetime(
                oof["label_resolution_available_at"], utc=True
            ),
        }
    )
    return candidate_identity, oof_lineage


def _write_canonical_packb_lineage(paths: dict[str, Path]) -> None:
    candidates, oof = _lineage_frames(paths)
    candidate_sides: dict[str, dict[str, str]] = {}
    residual_sides: dict[str, dict[str, object]] = {}
    for side in ("long", "short"):
        candidate_rows = candidates.loc[candidates["side_name"].eq(side)]
        oof_rows = oof.loc[oof["side_name"].eq(side)]
        packb = {
            "side": side,
            "source_hash": _lineage_hash(f"packb-source-{side}"),
            "model_hash": _lineage_hash(f"packb-model-{side}"),
            "feature_contract_hash": _lineage_hash(f"packb-features-{side}"),
            "parameter_hash": _lineage_hash(f"packb-parameters-{side}"),
            "candidate_row_identity_hash": materializer._row_identity_hash(
                candidate_rows
            ),
            "oof_row_identity_hash": materializer._row_identity_hash(oof_rows),
            "oof_fold_cutoff_hash": materializer._oof_fold_cutoff_hash(oof_rows),
        }
        candidate_sides[side] = packb
        residual_sides[side] = {
            "side": side,
            "source_hash": _lineage_hash(f"residual-source-{side}"),
            "model_hash": _lineage_hash(f"residual-model-{side}"),
            "feature_contract_hash": _lineage_hash(f"residual-features-{side}"),
            "parameter_hash": _lineage_hash(f"residual-parameters-{side}"),
            "oof_row_identity_hash": packb["oof_row_identity_hash"],
            "oof_fold_cutoff_hash": packb["oof_fold_cutoff_hash"],
            "upstream_packb": {"side": side, **packb},
        }

    candidate_manifest = {
        "schema": "packb_candidate_handoff_lineage_v1",
        "source_artifacts": {
            "candidate_handoff": {"sha256": materializer._sha256(paths["candidates"])}
        },
        "packb_per_side_lineage": {
            "model_side_scope": "per_side",
            "sides": candidate_sides,
        },
    }
    residual_manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    residual_manifest["source_artifacts"] = {
        "residual_oof": {"sha256": materializer._sha256(paths["oof"])}
    }
    residual_manifest["residual_per_side_lineage"] = {
        "model_side_scope": "per_side",
        "sides": residual_sides,
    }
    paths["candidate_manifest"].write_text(
        json.dumps(candidate_manifest), encoding="utf-8"
    )
    paths["manifest"].write_text(json.dumps(residual_manifest), encoding="utf-8")


def _inputs(tmp_path: Path) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    june = pd.Timestamp("2026-06-30T23:00:00Z")
    july_1 = pd.Timestamp("2026-07-01T00:00:00Z")
    july_2 = pd.Timestamp("2026-07-02T00:00:00Z")
    candidates = pd.DataFrame(
        {
            "__ts__": [june, june, july_1, july_1, july_2, july_2],
            "__symbol__": ["BTC", "ETH", "BTC", "ETH", "BTC", "ETH"],
            "side_name": ["LONG", "long", "long", "SHORT", "long", "short"],
            "candidate_id": [
                "june-btc",
                "june-eth",
                "july-1-btc",
                "july-1-eth",
                "july-2-btc",
                "july-2-eth",
            ],
            "base_leaf_bin": ["A", "A", "A", "B", "A", "B"],
            "meta_leaf_bin": ["x", "y", "x", "x", "x", "z"],
            "archetype_label_family": [
                "trend",
                "reversal",
                "trend",
                "reversal",
                "trend",
                "reversal",
            ],
            "archetype_policy_key": [
                "trend_fast",
                "reversal_slow",
                "trend_fast",
                "reversal_fast",
                "trend_slow",
                "reversal_slow",
            ],
            "available_at": [june, june, july_1, july_1, july_2, july_2],
            "unread_realized_outcome": [1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            "catboost_path_outcome_label": ["unused"] * 6,
        }
    )
    oof = pd.DataFrame(
        {
            "__ts__": [july_1, july_1, july_2, july_2],
            "__symbol__": ["BTC", "ETH", "BTC", "ETH"],
            "side_name": ["long", "short", "long", "short"],
            "candidate_id": ["july-1-btc", "july-1-eth", "july-2-btc", "july-2-eth"],
            "oof_fold": ["july_1", "july_1", "july_2", "july_2"],
            "validation_start": [july_1, july_1, july_2, july_2],
            "train_decision_cutoff": [
                july_1 - pd.Timedelta(hours=1),
                july_1 - pd.Timedelta(hours=1),
                july_2 - pd.Timedelta(hours=1),
                july_2 - pd.Timedelta(hours=1),
            ],
            "label_resolution_available_at": [
                july_1 - pd.Timedelta(hours=2),
                july_1 - pd.Timedelta(hours=2),
                july_2 - pd.Timedelta(hours=2),
                july_2 - pd.Timedelta(hours=2),
            ],
            "available_at": [july_1, july_1, july_2, july_2],
            "residual_ev": [0.04, -0.02, 0.03, 0.01],
            "base_ev": [0.01, -0.05, 0.02, 0.03],
            "unused_label": [1.0, 0.0, 1.0, 0.0],
        }
    )
    manifest = {
        "residual_expert_target": (
            "ev_after_1pct - train_only_hierarchical_expected_ev(base_score, side, archetype)"
        ),
        "folds": [
            {
                "fold_id": "july_1",
                "test_start": "2026-07-01T00:00:00Z",
                "test_end_exclusive": "2026-07-02T00:00:00Z",
            },
            {
                "fold_id": "july_2",
                "test_start": "2026-07-02T00:00:00Z",
                "test_end_exclusive": "2026-07-03T00:00:00Z",
            },
        ],
    }
    paths = {
        "oof": tmp_path / "residual_oof.parquet",
        "candidates": tmp_path / "candidate_handoff.parquet",
        "manifest": tmp_path / "residual_manifest.json",
        "candidate_manifest": tmp_path / "candidate_handoff.manifest.json",
    }
    oof.to_parquet(paths["oof"], index=False)
    candidates.to_parquet(paths["candidates"], index=False)
    paths["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    _write_canonical_packb_lineage(paths)
    return paths


def _args(
    tmp_path: Path, paths: dict[str, Path], **overrides: object
) -> SimpleNamespace:
    values: dict[str, object] = {
        "residual_oof": paths["oof"],
        "candidate_handoff": paths["candidates"],
        "candidate_manifest": paths["candidate_manifest"],
        "residual_manifest": paths["manifest"],
        "lineage_mode": "canonical_packb",
        "output": tmp_path / "alpha_oof.parquet",
        "output_manifest": tmp_path / "alpha_oof.manifest.json",
        "residual_ev_col": "residual_ev",
        "base_ev_col": "base_ev",
        "leaf_bin_cols": ["base_leaf_bin", "meta_leaf_bin"],
        "base_archetype_source_cols": [
            "archetype_label_family",
            "archetype_policy_key",
        ],
        "base_archetype_canonical_source": "archetype_label_family",
        "oof_timestamp_col": "__ts__",
        "oof_symbol_col": "__symbol__",
        "oof_side_col": "side_name",
        "oof_candidate_id_col": "candidate_id",
        "oof_fold_col": "oof_fold",
        "oof_validation_start_col": "validation_start",
        "oof_train_decision_cutoff_col": "train_decision_cutoff",
        "oof_label_resolution_available_at_col": "label_resolution_available_at",
        "oof_available_at_col": "available_at",
        "candidate_timestamp_col": "__ts__",
        "candidate_symbol_col": "__symbol__",
        "candidate_side_col": "side_name",
        "candidate_id_col": "candidate_id",
        "candidate_available_at_col": "available_at",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_materializes_exact_oof_alpha_and_fold_causal_leaf_support(
    tmp_path: Path,
) -> None:
    paths = _inputs(tmp_path)
    result = materializer.run(_args(tmp_path, paths))
    output = pd.read_parquet(result["output"])
    manifest = json.loads(result["manifest"].read_text(encoding="utf-8"))

    assert output["side_name"].tolist() == ["long", "short", "long", "short"]
    assert output["candidate_id"].tolist() == [
        "july-1-btc",
        "july-1-eth",
        "july-2-btc",
        "july-2-eth",
    ]
    assert output["oof_fold"].tolist() == ["july_1", "july_1", "july_2", "july_2"]
    assert output["validation_start"].tolist() == [
        pd.Timestamp("2026-07-01T00:00:00Z"),
        pd.Timestamp("2026-07-01T00:00:00Z"),
        pd.Timestamp("2026-07-02T00:00:00Z"),
        pd.Timestamp("2026-07-02T00:00:00Z"),
    ]
    assert (output["train_decision_cutoff"] < output["validation_start"]).all()
    assert (
        output["label_resolution_available_at"] <= output["train_decision_cutoff"]
    ).all()
    assert output["alpha_prediction_uncertainty"].tolist() == pytest.approx(
        [0.03, 0.03, 0.01, 0.02]
    )
    btc_july_1 = output.loc[
        output["__ts__"].eq(pd.Timestamp("2026-07-01T00:00:00Z"))
        & output["__symbol__"].eq("BTC")
    ].iloc[0]
    assert btc_july_1["alpha_leaf_tuple_support_log1p"] == pytest.approx(np.log1p(1))
    assert btc_july_1["alpha_leaf_individual_support_log1p_min"] == pytest.approx(
        np.log1p(1)
    )
    assert btc_july_1["alpha_leaf_support"] == pytest.approx(np.log1p(1))
    btc_july_2 = output.loc[
        output["__ts__"].eq(pd.Timestamp("2026-07-02T00:00:00Z"))
        & output["__symbol__"].eq("BTC")
    ].iloc[0]
    assert btc_july_2["alpha_leaf_support"] == pytest.approx(np.log1p(2))
    eth_july_1 = output.loc[
        output["__ts__"].eq(pd.Timestamp("2026-07-01T00:00:00Z"))
        & output["__symbol__"].eq("ETH")
    ].iloc[0]
    assert eth_july_1["alpha_leaf_support"] == 0.0
    assert output["available_at"].eq(output["__ts__"]).all()
    assert "unused_label" not in output.columns
    assert "unread_realized_outcome" not in output.columns
    assert "catboost_path_outcome_label" not in output.columns
    assert "archetype_label_family" not in output.columns
    assert "archetype_policy_key" not in output.columns
    archetype_contract = manifest["definitions"][
        "base_archetype_label_feature_contract"
    ]
    candidate_frame = pd.read_parquet(paths["candidates"])
    expected_archetype_contract = (
        materializer.fit_base_archetype_label_feature_contract(
            candidate_frame,
            source_columns=["archetype_label_family", "archetype_policy_key"],
            canonical_source="archetype_label_family",
        )
    )
    assert archetype_contract == expected_archetype_contract
    archetype_features = list(archetype_contract["features"])
    assert archetype_contract["source_columns"] == [
        "archetype_label_family",
        "archetype_policy_key",
    ]
    assert archetype_contract["canonical_source"] == "archetype_label_family"
    assert all(name.startswith("base_archetype_label__") for name in archetype_features)
    assert all(
        output[name].dtype == np.dtype(np.float32) for name in archetype_features
    )
    assert not any("catboost" in name or "path" in name for name in archetype_features)
    expected_onehots = materializer.transform_base_archetype_label_features(
        candidate_frame,
        expected_archetype_contract,
    )
    expected_btc_july_1 = expected_onehots.iloc[2]
    actual_btc_july_1 = output.loc[
        output["__ts__"].eq(pd.Timestamp("2026-07-01T00:00:00Z"))
        & output["__symbol__"].eq("BTC"),
        archetype_features,
    ].iloc[0]
    np.testing.assert_array_equal(
        actual_btc_july_1.to_numpy(dtype=np.float32),
        expected_btc_july_1.to_numpy(dtype=np.float32),
    )
    assert manifest["schema"] == "execution_ev_alpha_oof_v3"
    assert manifest["definitions"]["outcome_contract"].startswith("no outcome")
    assert manifest["source_artifacts"]["residual_oof"]["sha256"]
    assert manifest["output_sha256"] == materializer._sha256(result["output"])
    assert manifest["source_artifact_sha256"] == materializer._sha256(result["output"])
    assert manifest["prediction_role"] == "alpha_ev_oof"
    assert manifest["prediction_columns"]["existing_alpha_ev"] == {
        "role": "pre_entry_alpha_ev_oof_prediction",
        "target": False,
    }
    assert manifest["lineage"]["mode"] == "canonical_packb"
    assert manifest["lineage"]["canonical"] is True
    assert set(manifest["lineage"]["per_side"]) == {"long", "short"}
    _, oof_lineage = _lineage_frames(paths)
    assert manifest["lineage"]["per_side"]["long"][
        "oof_row_identity_hash"
    ] == materializer._row_identity_hash(
        oof_lineage.loc[oof_lineage["side_name"].eq("long")]
    )
    assert manifest["alpha_cost_basis"]["deducted_cost_return"] == pytest.approx(0.01)
    assert (
        manifest["alpha_cost_basis"]["target_semantics"] == "residual_net_ev_after_1pct"
    )
    assert manifest[
        "prediction_role_manifest_sha256"
    ] == materializer._canonical_json_hash(
        manifest, excluded=("prediction_role_manifest_sha256",)
    )


def test_accepts_prediction_available_at_execution_decision(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    oof = pd.read_parquet(paths["oof"])
    oof["available_at"] = pd.to_datetime(oof["__ts__"], utc=True) + pd.Timedelta(
        hours=1
    )
    oof.to_parquet(paths["oof"], index=False)
    _write_canonical_packb_lineage(paths)

    result = materializer.run(_args(tmp_path, paths))
    output = pd.read_parquet(result["output"])
    assert (
        pd.to_datetime(output["available_at"], utc=True)
        == pd.to_datetime(output["__ts__"], utc=True) + pd.Timedelta(hours=1)
    ).all()


def test_leaf_support_excludes_future_candidate_rows(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    candidates = pd.read_parquet(paths["candidates"])
    future = candidates.iloc[[0]].copy()
    future["__ts__"] = pd.Timestamp("2026-07-10T00:00:00Z")
    candidates = pd.concat([candidates, future], ignore_index=True)
    candidates.to_parquet(paths["candidates"], index=False)
    _write_canonical_packb_lineage(paths)

    output = pd.read_parquet(materializer.run(_args(tmp_path, paths))["output"])
    btc_july_1 = output.loc[
        output["__ts__"].eq(pd.Timestamp("2026-07-01T00:00:00Z"))
        & output["__symbol__"].eq("BTC")
    ].iloc[0]
    assert btc_july_1["alpha_leaf_support"] == pytest.approx(np.log1p(1))


def test_canonical_packb_lineage_requires_both_bound_manifests(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)

    with pytest.raises(ValueError, match="requires --candidate-manifest"):
        materializer.run(_args(tmp_path, paths, candidate_manifest=None))

    candidate_manifest = json.loads(
        paths["candidate_manifest"].read_text(encoding="utf-8")
    )
    candidate_manifest["packb_per_side_lineage"]["sides"]["long"]["model_hash"] = (
        candidate_manifest["packb_per_side_lineage"]["sides"]["short"]["model_hash"]
    )
    paths["candidate_manifest"].write_text(
        json.dumps(candidate_manifest), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="must not share fitted model_hash"):
        materializer.run(_args(tmp_path, paths))


def test_canonical_packb_lineage_rejects_mismatched_residual_upstream_hash(
    tmp_path: Path,
) -> None:
    paths = _inputs(tmp_path)
    residual_manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    residual_manifest["residual_per_side_lineage"]["sides"]["short"]["upstream_packb"][
        "feature_contract_hash"
    ] = _lineage_hash("wrong-packb-feature-contract")
    paths["manifest"].write_text(json.dumps(residual_manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="upstream Pack-B feature_contract_hash"):
        materializer.run(_args(tmp_path, paths))


def test_canonical_packb_lineage_binds_the_supplied_candidate_artifact(
    tmp_path: Path,
) -> None:
    paths = _inputs(tmp_path)
    candidates = pd.read_parquet(paths["candidates"])
    candidates.loc[0, "base_leaf_bin"] = "changed-after-manifest"
    candidates.to_parquet(paths["candidates"], index=False)

    with pytest.raises(ValueError, match="does not bind the supplied artifact"):
        materializer.run(_args(tmp_path, paths))


def test_canonical_packb_lineage_recomputes_oof_fold_cutoff_provenance(
    tmp_path: Path,
) -> None:
    paths = _inputs(tmp_path)
    oof = pd.read_parquet(paths["oof"])
    oof.loc[0, "train_decision_cutoff"] = pd.Timestamp("2026-06-30T22:00:00Z")
    oof.to_parquet(paths["oof"], index=False)
    residual_manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    residual_manifest["source_artifacts"]["residual_oof"]["sha256"] = (
        materializer._sha256(paths["oof"])
    )
    paths["manifest"].write_text(json.dumps(residual_manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="oof fold/cutoff hash does not match"):
        materializer.run(_args(tmp_path, paths))


def test_historical_comparator_mode_is_explicitly_non_canonical(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    result = materializer.run(
        _args(
            tmp_path,
            paths,
            candidate_manifest=None,
            lineage_mode="historical_comparator",
        )
    )
    manifest = json.loads(result["manifest"].read_text(encoding="utf-8"))

    assert manifest["lineage"] == {
        "mode": "historical_comparator",
        "canonical": False,
        "reason": (
            "explicit historical-comparator mode; Pack-B per-side lineage was not "
            "required and this output must not be used as canonical downstream input"
        ),
        "candidate_manifest": None,
    }


@pytest.mark.parametrize(
    "mutation, error", [("duplicate", "duplicate rows"), ("nonfinite", "non-finite")]
)
def test_rejects_duplicate_or_nonfinite_required_rows(
    tmp_path: Path, mutation: str, error: str
) -> None:
    paths = _inputs(tmp_path)
    if mutation == "duplicate":
        oof = pd.read_parquet(paths["oof"])
        pd.concat([oof, oof.iloc[[0]]], ignore_index=True).to_parquet(
            paths["oof"], index=False
        )
    else:
        oof = pd.read_parquet(paths["oof"])
        oof.loc[0, "residual_ev"] = np.inf
        oof.to_parquet(paths["oof"], index=False)
    with pytest.raises(ValueError, match=error):
        materializer.run(_args(tmp_path, paths))


def test_rejects_missing_exact_candidate_handoff_row(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    candidates = pd.read_parquet(paths["candidates"])
    candidates.drop(index=2).to_parquet(paths["candidates"], index=False)
    _write_canonical_packb_lineage(paths)

    with pytest.raises(ValueError, match="no exact candidate identity match"):
        materializer.run(_args(tmp_path, paths))


@pytest.mark.parametrize(
    "column",
    [
        "candidate_id",
        "oof_fold",
        "validation_start",
        "train_decision_cutoff",
        "label_resolution_available_at",
        "available_at",
    ],
)
def test_rejects_missing_canonical_alpha_provenance_columns(
    tmp_path: Path, column: str
) -> None:
    paths = _inputs(tmp_path)
    oof = pd.read_parquet(paths["oof"]).drop(columns=column)
    oof.to_parquet(paths["oof"], index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        materializer.run(_args(tmp_path, paths))


@pytest.mark.parametrize(
    "column, value, error",
    [
        ("candidate_id", "wrong-candidate", "no exact candidate identity match"),
        ("oof_fold", "unknown", "not declared by residual_manifest"),
        (
            "validation_start",
            pd.Timestamp("2026-06-30T23:30:00Z"),
            "does not match manifest",
        ),
        (
            "train_decision_cutoff",
            pd.Timestamp("2026-07-01T00:00:00Z"),
            "strictly before validation start",
        ),
        (
            "label_resolution_available_at",
            pd.Timestamp("2026-07-01T00:00:00Z"),
            "training labels must resolve before",
        ),
        (
            "available_at",
            pd.Timestamp("2026-07-01T02:00:00Z"),
            "feature availability is after",
        ),
    ],
)
def test_rejects_invalid_canonical_alpha_provenance(
    tmp_path: Path, column: str, value: object, error: str
) -> None:
    paths = _inputs(tmp_path)
    oof = pd.read_parquet(paths["oof"])
    oof.loc[0, column] = value
    oof.to_parquet(paths["oof"], index=False)
    if column in {"candidate_id", "oof_fold", "validation_start"}:
        _write_canonical_packb_lineage(paths)

    with pytest.raises(ValueError, match=error):
        materializer.run(_args(tmp_path, paths))


def test_rejects_rows_outside_manifest_boundaries_and_late_leaf_availability(
    tmp_path: Path,
) -> None:
    paths = _inputs(tmp_path)
    oof = pd.read_parquet(paths["oof"])
    oof.loc[0, "__ts__"] = pd.Timestamp("2026-07-04T00:00:00Z")
    oof.to_parquet(paths["oof"], index=False)
    candidates = pd.read_parquet(paths["candidates"])
    candidates.loc[2, "__ts__"] = pd.Timestamp("2026-07-04T00:00:00Z")
    candidates.loc[2, "available_at"] = pd.Timestamp("2026-07-04T00:00:00Z")
    candidates.to_parquet(paths["candidates"], index=False)
    _write_canonical_packb_lineage(paths)
    with pytest.raises(ValueError, match="conflicts with manifest boundaries"):
        materializer.run(_args(tmp_path, paths))

    paths = _inputs(tmp_path / "late")
    candidates = pd.read_parquet(paths["candidates"])
    candidates.loc[0, "available_at"] = candidates.loc[0, "__ts__"] + pd.Timedelta(
        hours=2
    )
    candidates.to_parquet(paths["candidates"], index=False)
    with pytest.raises(ValueError, match="availability is after"):
        materializer.run(_args(tmp_path / "late", paths))


@pytest.mark.parametrize(
    "column, value, error",
    [
        ("archetype_label_family", "", "null or blank identity values"),
        ("archetype_policy_key", None, "null or blank identity values"),
    ],
)
def test_rejects_incomplete_base_archetype_source_values(
    tmp_path: Path, column: str, value: object, error: str
) -> None:
    paths = _inputs(tmp_path)
    candidates = pd.read_parquet(paths["candidates"])
    candidates.loc[0, column] = value
    candidates.to_parquet(paths["candidates"], index=False)

    with pytest.raises(ValueError, match=error):
        materializer.run(_args(tmp_path, paths))


@pytest.mark.parametrize(
    "source_column", ["catboost_base_archetype", "path_outcome_label"]
)
def test_rejects_catboost_or_path_outcome_archetype_sources(
    tmp_path: Path, source_column: str
) -> None:
    paths = _inputs(tmp_path)
    with pytest.raises(
        ValueError, match="CatBoost and path outcome labels are forbidden"
    ):
        materializer.run(
            _args(
                tmp_path,
                paths,
                base_archetype_source_cols=[source_column],
                base_archetype_canonical_source=source_column,
            )
        )
