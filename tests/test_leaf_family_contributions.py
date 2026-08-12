from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.leaf_family_contributions import (
    LeafFamilyContributionConfig,
    LeafFamilyContributionError,
    VALUE_COLUMN,
    _iter_assignment_batches,
    extract_leaf_family_contributions,
    materialize_leaf_family_contributions,
)


def _write_artifact(
    root: Path,
    *,
    unknown_token: bool = False,
    assignment_identity_mismatch: bool = False,
    bad_additive_value: bool = False,
) -> Path:
    artifact = root / "strict_artifact"
    artifact.mkdir(parents=True)
    timestamp = pd.to_datetime(["2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z"], utc=True)
    assignments = pd.DataFrame({
        "candidate_id": ["candidate-a", "candidate-b"],
        "__ts__": timestamp,
        "side_name": ["long", "long"],
        "head_name": ["p_clear", "p_clear"],
        "fold_id": ["fold-1", "fold-1"],
        "leaf_assignment__model_00_head_tree_000": np.array([101, 102], dtype=np.uint64),
        "leaf_assignment__model_01_head_tree_000": np.array([201, 202], dtype=np.uint64),
    })
    if unknown_token:
        assignments.loc[1, "leaf_assignment__model_01_head_tree_000"] = np.uint64(999)
    if assignment_identity_mismatch:
        # A separate parquet column cannot have a distinct identity row, so
        # emulate an artifact field that violates the manifest scope.
        assignments.loc[1, "fold_id"] = "another-fold"
    assignments.to_parquet(artifact / "leaf_assignments.parquet", index=False)
    second_contribution = -1.0 if not bad_additive_value else -1.25
    catalog = pd.DataFrame({
        "head_name": ["p_clear"] * 4,
        "side_name": ["long"] * 4,
        "fold_id": ["fold-1"] * 4,
        "model_slot": [0, 0, 1, 1],
        "head_tree_slot": [0, 0, 0, 0],
        "leaf_token": np.array([101, 102, 201, 202], dtype=np.uint64),
        "rule_signature": ["rule-alpha", "rule-beta", "rule-alpha", "rule-gamma"],
        "tree_leaf_value": [2.0, -4.0, 6.0, -2.0],
        # Two fitted base models: every selected tree leaf value is divided by 2.
        "ensemble_tree_contribution": [1.0, -2.0, 3.0, second_contribution],
    })
    catalog.to_parquet(artifact / "leaf_rule_catalog.parquet", index=False)
    (artifact / "base_reasoning_manifest.json").write_text(json.dumps({
        "schema": "strict_oof_base_reasoning_v2",
        "status": "MATERIALIZED_STRICT_OOF",
        "head_name": "p_clear",
        "side_name": "long",
        "fold_id": "fold-1",
        "provenance": {"model_hashes": ["model-a", "model-b"]},
    }))
    return artifact


def test_extracts_same_artifact_additive_family_contributions_without_tokens(tmp_path: Path) -> None:
    artifact = _write_artifact(tmp_path)

    result = extract_leaf_family_contributions(
        artifact,
        config=LeafFamilyContributionConfig(assignment_batch_rows=1, max_rows_per_output_bucket=1),
    ).sort_values(["candidate_id", "rule_signature"], kind="stable").reset_index(drop=True)

    expected = pd.DataFrame({
        "candidate_id": ["candidate-a", "candidate-b", "candidate-b"],
        "rule_signature": ["rule-alpha", "rule-beta", "rule-gamma"],
        "contribution_direction": ["positive", "negative", "negative"],
        VALUE_COLUMN: [4.0, -2.0, -1.0],
    })
    assert result[["candidate_id", "rule_signature", "contribution_direction", VALUE_COLUMN]].equals(expected)
    assert result["__ts__"].dt.tz is not None
    assert result["side_name"].eq("long").all()
    assert result["fold_id"].eq("fold-1").all()
    assert result["head_name"].eq("p_clear").all()
    assert not any(
        forbidden in column.lower()
        for column in result
        for forbidden in ("leaf_token", "leaf_id", "leaf_assignment", "raw_leaf")
    )
    # The output is candidate/family additive, including all selected trees.
    reconstructed = result.groupby("candidate_id", sort=False)[VALUE_COLUMN].sum().to_dict()
    assert reconstructed == {"candidate-a": 4.0, "candidate-b": -3.0}


def test_writes_immutable_token_free_table(tmp_path: Path) -> None:
    artifact = _write_artifact(tmp_path)
    output = tmp_path / "family_contributions.parquet"

    materialized = materialize_leaf_family_contributions(
        artifact,
        output,
        config=LeafFamilyContributionConfig(assignment_batch_rows=1, max_rows_per_output_bucket=1),
    )

    assert materialized.candidate_count == 2
    assert materialized.assignment_column_count == 2
    assert materialized.contribution_row_count == 3
    assert materialized.family_contribution_total == pytest.approx(1.0)
    assert output.is_file()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        materialize_leaf_family_contributions(artifact, output)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"unknown_token": True}, "absent from its same-artifact catalog"),
        ({"assignment_identity_mismatch": True}, "crosses the artifact fold scope"),
        ({"bad_additive_value": True}, "does not reconcile"),
    ],
)
def test_rejects_cross_scope_or_unreconciled_same_artifact_joins(
    tmp_path: Path,
    kwargs: dict[str, bool],
    message: str,
) -> None:
    artifact = _write_artifact(tmp_path, **kwargs)
    with pytest.raises(LeafFamilyContributionError, match=message):
        extract_leaf_family_contributions(artifact)


def test_assignment_reader_is_one_tree_column_bounded() -> None:
    """Guard against regressing to a full-wide assignment parquet read."""

    source = inspect.getsource(_iter_assignment_batches)
    assert "iter_batches" in source
    assert "assignment_column" in source
    assert "columns = [*IDENTITY, *SCOPE, assignment_column]" in source
