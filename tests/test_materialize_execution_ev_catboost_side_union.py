from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "materialize_execution_ev_catboost_side_union",
    ROOT / "scripts" / "materialize_execution_ev_catboost_side_union.py",
)
assert SPEC and SPEC.loader
materializer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = materializer
SPEC.loader.exec_module(materializer)


def _side(directory: Path, side: str) -> None:
    directory.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(
                ["2026-05-01T00:00:00Z", "2026-06-01T00:00:00Z"], utc=True
            ),
            "__symbol__": ["BTC", "ETH"],
            "side_name": [side, side],
            "candidate_id": [f"{side}-1", f"{side}-2"],
            "oof_fold_id": [0, 1],
            "validation_start": pd.to_datetime(
                ["2026-05-01T00:00:00Z", "2026-06-01T00:00:00Z"], utc=True
            ),
            "train_decision_cutoff": pd.to_datetime(
                ["2026-04-30T00:00:00Z", "2026-05-31T00:00:00Z"], utc=True
            ),
            "label_resolution_available_at": pd.to_datetime(
                ["2026-04-30T00:00:00Z", "2026-05-31T00:00:00Z"], utc=True
            ),
            "available_at": pd.to_datetime(
                ["2026-05-01T00:00:00Z", "2026-06-01T00:00:00Z"], utc=True
            ),
            "predicted_path_archetype": [
                materializer.CLASS_ORDER[0],
                materializer.CLASS_ORDER[1],
            ],
            "probability_entropy": [1.0, 1.1],
            "max_probability": [0.4, 0.35],
            "normalized_entropy": [0.5, 0.6],
            "top2_probability_margin": [0.1, 0.05],
            "adverse_probability_mass": [0.5, 0.45],
            "favorable_probability_mass": [0.4, 0.45],
            **{
                f"probability__{class_name}": [1.0 / 7.0, 1.0 / 7.0]
                for class_name in materializer.CLASS_ORDER
            },
        }
    )
    parquet = directory / materializer.OOF_PARQUET
    frame.to_parquet(parquet, index=False)
    prediction_columns = {
        column: {
            "role": "pre_entry_path_archetype_oof_prediction",
            "target": False,
        }
        for column in frame
        if column.startswith("probability__")
        or column
        in {
            "predicted_path_archetype",
            "probability_entropy",
            "max_probability",
            "normalized_entropy",
            "top2_probability_margin",
            "adverse_probability_mass",
            "favorable_probability_mass",
        }
    }
    manifest = {
        "prediction_role": materializer.PREDICTION_ROLE,
        "prediction_columns": prediction_columns,
        "source_artifact_sha256": materializer._sha256(parquet),
    }
    manifest["prediction_role_manifest_sha256"] = materializer._canonical_json_hash(
        manifest
    )
    (directory / materializer.ROLE_MANIFEST).write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def _args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        long_dir=tmp_path / "long",
        short_dir=tmp_path / "short",
        output=tmp_path / "union.parquet",
        manifest=tmp_path / "union.manifest.json",
    )


def test_materializes_signed_disjoint_per_side_union(tmp_path: Path) -> None:
    args = _args(tmp_path)
    _side(args.long_dir, "long")
    _side(args.short_dir, "short")

    outputs = materializer.run(args)
    union = pd.read_parquet(outputs["output"])
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert len(union) == 4
    assert union["side_name"].value_counts().to_dict() == {"long": 2, "short": 2}
    assert manifest["model_side_scope"] == "per_side"
    assert manifest["shared_fitted_state"] is False
    assert manifest["class_names"] == list(materializer.CLASS_ORDER)
    assert manifest["source_artifact_sha256"] == materializer._sha256(outputs["output"])
    assert manifest[
        "prediction_role_manifest_sha256"
    ] == materializer._canonical_json_hash(
        manifest, excluded=("prediction_role_manifest_sha256",)
    )


def test_rejects_wrong_side_contents(tmp_path: Path) -> None:
    args = _args(tmp_path)
    _side(args.long_dir, "short")
    _side(args.short_dir, "short")
    with pytest.raises(ValueError, match="wrong side"):
        materializer.run(args)
