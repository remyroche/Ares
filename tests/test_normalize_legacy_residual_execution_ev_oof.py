from __future__ import annotations

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
    "normalize_legacy_residual_execution_ev_oof",
    ROOT / "scripts" / "normalize_legacy_residual_execution_ev_oof.py",
)
assert SPEC and SPEC.loader
normalizer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = normalizer
SPEC.loader.exec_module(normalizer)


def _inputs(tmp_path: Path) -> dict[str, Path]:
    march_valid = pd.Timestamp("2026-03-30T19:00:00Z")
    march_purged = pd.Timestamp("2026-03-30T21:00:00Z")
    april_test = pd.Timestamp("2026-04-01T00:00:00Z")
    april_valid = pd.Timestamp("2026-04-29T19:00:00Z")
    april_purged = pd.Timestamp("2026-04-29T21:00:00Z")
    may_test = pd.Timestamp("2026-05-01T00:00:00Z")
    handoff = pd.DataFrame(
        {
            "__ts__": [
                march_valid,
                march_purged,
                april_test,
                april_valid,
                april_purged,
                may_test,
            ],
            "__symbol__": ["BTC", "ETH", "BTC", "BTC", "ETH", "ETH"],
            "side_name": ["LONG", "short", "long", "long", "short", "short"],
            "__label_path_end_ts__": [
                pd.Timestamp("2026-03-31T21:00:00Z"),
                pd.Timestamp("2026-03-31T23:00:00Z"),
                pd.Timestamp("2026-04-02T00:00:00Z"),
                pd.Timestamp("2026-04-30T21:00:00Z"),
                pd.Timestamp("2026-04-30T23:00:00Z"),
                pd.Timestamp("2026-05-02T00:00:00Z"),
            ],
            "archetype_label_family": ["trend"] * 6,
            "archetype_policy_key": ["trend_fast"] * 6,
            "base_leaf_bin": ["a", "b", "a", "a", "b", "b"],
            "meta_leaf_bin": ["x", "y", "x", "x", "y", "y"],
        }
    )
    oof = pd.DataFrame(
        {
            "__ts__": [april_test, may_test],
            "__symbol__": ["BTC", "ETH"],
            "side_name": ["long", "SHORT"],
            # This is the known legacy host-timezone cast: exactly +2h.
            "__label_path_end_ts__": [
                pd.Timestamp("2026-04-02T02:00:00Z"),
                pd.Timestamp("2026-05-02T02:00:00Z"),
            ],
            "score_base_ev_residual_expert_hier_mapped": np.array(
                [0.03125, -0.0625], dtype=np.float32
            ),
            "score_base_ev_mapped": np.array([0.01, -0.05], dtype=np.float32),
        }
    )
    manifest = {
        "residual_expert_target": (
            "ev_after_1pct - train-only hierarchical expected EV(base_score, side, archetype)"
        ),
        "folds": [
            {
                "train_end_exclusive": "2026-04-01T00:00:00Z",
                "test_start": "2026-04-01T00:00:00Z",
                "test_end_exclusive": "2026-05-01T00:00:00Z",
                "train_rows": 1,
                "test_rows": 1,
            },
            {
                "train_end_exclusive": "2026-05-01T00:00:00Z",
                "test_start": "2026-05-01T00:00:00Z",
                "test_end_exclusive": "2026-06-01T00:00:00Z",
                "train_rows": 4,
                "test_rows": 1,
            },
        ]
    }
    paths = {
        "oof": tmp_path / "legacy_oof.parquet",
        "handoff": tmp_path / "handoff.parquet",
        "manifest": tmp_path / "residual_manifest.json",
    }
    oof.to_parquet(paths["oof"], index=False)
    handoff.to_parquet(paths["handoff"], index=False)
    paths["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    return paths


def _args(tmp_path: Path, paths: dict[str, Path], **overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "residual_oof": paths["oof"],
        "candidate_handoff": paths["handoff"],
        "residual_manifest": paths["manifest"],
        "output_oof": tmp_path / "normalized_oof.parquet",
        "output_candidate_handoff": tmp_path / "normalized_handoff.parquet",
        "output_manifest": tmp_path / "normalized.provenance.json",
        "residual_ev_col": "score_base_ev_residual_expert_hier_mapped",
        "base_ev_col": "score_base_ev_mapped",
        "residual_label_end_col": "__label_path_end_ts__",
        "handoff_label_end_col": "__label_path_end_ts__",
        "legacy_label_end_offset_hours": 2.0,
        "candidate_columns": [
            "archetype_label_family",
            "archetype_policy_key",
            "base_leaf_bin",
            "meta_leaf_bin",
        ],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_normalizes_legacy_oof_with_exact_identity_and_fold_evidence(
    tmp_path: Path,
) -> None:
    paths = _inputs(tmp_path)
    result = normalizer.run(_args(tmp_path, paths))
    normalized = pd.read_parquet(result["oof"])
    candidates = pd.read_parquet(result["candidate_handoff"])
    manifest = json.loads(result["manifest"].read_text(encoding="utf-8"))
    source = pd.read_parquet(paths["oof"])

    assert normalized["candidate_id"].tolist() == [
        "BTC|2026-04-01T00:00:00Z|1h|long",
        "ETH|2026-05-01T00:00:00Z|1h|short",
    ]
    assert candidates["candidate_id"].is_unique
    assert normalized["oof_fold"].tolist() == ["0", "1"]
    assert normalized["validation_start"].tolist() == [
        pd.Timestamp("2026-04-01T00:00:00Z"),
        pd.Timestamp("2026-05-01T00:00:00Z"),
    ]
    assert normalized["train_max_decision_ts"].tolist() == [
        pd.Timestamp("2026-03-30T19:00:00Z"),
        pd.Timestamp("2026-04-29T19:00:00Z"),
    ]
    assert normalized["train_decision_cutoff"].tolist() == [
        pd.Timestamp("2026-03-31T23:00:00Z"),
        pd.Timestamp("2026-04-30T23:00:00Z"),
    ]
    assert normalized["label_resolution_available_at"].eq(
        normalized["train_decision_cutoff"]
    ).all()
    assert (normalized["train_decision_cutoff"] < normalized["validation_start"]).all()
    assert normalized["available_at"].eq(normalized["__ts__"]).all()
    assert np.array_equal(
        normalized["score_base_ev_residual_expert_hier_mapped"].to_numpy(),
        source["score_base_ev_residual_expert_hier_mapped"].to_numpy(),
    )
    assert np.array_equal(
        normalized["score_base_ev_mapped"].to_numpy(),
        source["score_base_ev_mapped"].to_numpy(),
    )
    assert manifest["legacy_label_resolution_contract"]["signed_offset_hours"] == 2.0
    assert manifest["alpha_cost_basis"] == {
        "deducted_cost_return": 0.01,
        "cost_unit": "return",
        "target_semantics": "residual_net_ev_after_1pct",
        "source_manifest": {
            "path": str(paths["manifest"].resolve()),
            "sha256": normalizer._sha256(paths["manifest"]),
        },
        "source_manifest_evidence": [
            {
                "field": "residual_expert_target",
                "value": (
                    "ev_after_1pct - train-only hierarchical expected EV(base_score, side, archetype)"
                ),
            }
        ],
        "verification": (
            "source manifest explicitly declares a residual target in ev_after_1pct "
            "units; that target contains exactly one 1% round-trip cost"
        ),
    }
    assert [fold["verified_train_rows"] for fold in manifest["fold_provenance"]] == [1, 4]
    assert manifest["provenance_manifest_sha256"]


def test_fails_closed_when_signed_offset_cannot_be_proven(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    corrupted = pd.read_parquet(paths["oof"])
    corrupted.loc[1, "__label_path_end_ts__"] = pd.Timestamp("2026-05-02T01:00:00Z")
    corrupted.to_parquet(paths["oof"], index=False)

    with pytest.raises(ValueError, match="label-resolution offset"):
        normalizer.run(_args(tmp_path, paths))


def test_fails_closed_when_manifest_train_count_is_not_reproducible(
    tmp_path: Path,
) -> None:
    paths = _inputs(tmp_path)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    manifest["folds"][0]["train_rows"] = 2
    paths["manifest"].write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="reconstructed train rows=1"):
        normalizer.run(_args(tmp_path, paths))


@pytest.mark.parametrize(
    ("target_evidence", "error"),
    [
        (None, "explicit 1% residual-net target evidence is required"),
        ("gross_ev_before_cost", "not an explicit residual ev_after_1pct target"),
    ],
)
def test_fails_closed_without_proven_legacy_alpha_cost_basis(
    tmp_path: Path, target_evidence: str | None, error: str
) -> None:
    paths = _inputs(tmp_path)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    if target_evidence is None:
        manifest.pop("residual_expert_target")
    else:
        manifest["residual_expert_target"] = target_evidence
    paths["manifest"].write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=error):
        normalizer.run(_args(tmp_path, paths))
