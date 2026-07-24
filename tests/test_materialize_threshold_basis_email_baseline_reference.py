from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "materialize_threshold_basis_email_baseline_reference.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("threshold_email_baseline", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_materialize_keeps_canonical_policy_alias_in_sync(tmp_path: Path) -> None:
    module = _load_module()
    artifact_root = tmp_path / "artifact"
    policy_dir = artifact_root / "policy_params"
    policy_dir.mkdir(parents=True)
    policy_path = policy_dir / "threshold_basis_policy_sidearch_ev70_trim10_21d.json"
    alias_path = policy_dir / "threshold_basis_policy.json"
    reference_path = policy_dir / "reference.parquet"
    source_path = tmp_path / "source.parquet"

    reference = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-07-01T00:00:00Z"]),
            "symbol": ["BTC/USD:USD"],
            "side_name": ["long"],
            "policy_archetype": ["breakout"],
            "expected_ev": [0.01],
        }
    )
    reference.to_parquet(reference_path, index=False)
    source = pd.DataFrame(
        {
            "__ts__": reference["timestamp"],
            "__symbol__": reference["symbol"],
            "side_name": reference["side_name"],
            "archetype_policy_key": ["long__breakout"],
            "clean_exec": [1.0],
        }
    )
    source.to_parquet(source_path, index=False)
    initial_policy = {
        "policy_id": "side_archetype_hier_ev_fixed70_trim10_21d_v1",
        "reference_candidates_path": reference_path.name,
    }
    policy_path.write_text(json.dumps(initial_policy), encoding="utf-8")
    alias_path.write_text(json.dumps({"stale": True}), encoding="utf-8")

    result = module.materialize(
        policy_path=policy_path,
        matrix_sources=[source_path],
        output_name="reference_enriched.parquet",
    )

    assert policy_path.read_bytes() == alias_path.read_bytes()
    assert result["synchronized_policy_aliases"] == [str(alias_path.resolve())]
    enriched = json.loads(alias_path.read_text(encoding="utf-8"))
    assert enriched["reference_candidates_path"] == "reference_enriched.parquet"
    assert enriched["email_archetype_baseline_diagnostics"] == ["clean_exec"]


def test_materialize_adds_exact_successful_trade_mae_from_labels(tmp_path: Path) -> None:
    module = _load_module()
    artifact_root = tmp_path / "artifact"
    policy_dir = artifact_root / "policy_params"
    policy_dir.mkdir(parents=True)
    policy_path = policy_dir / "threshold_basis_policy.json"
    reference_path = policy_dir / "reference.parquet"
    source_path = tmp_path / "source.parquet"
    label_root = tmp_path / "labels"
    label_root.mkdir()

    reference = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-07-01T00:00:00Z"]),
            "symbol": ["BTC/USD:USD"],
            "side_name": ["long"],
            "policy_archetype": ["breakout"],
            "expected_ev": [0.01],
        }
    )
    reference.to_parquet(reference_path, index=False)
    pd.DataFrame(
        {
            "__ts__": reference["timestamp"],
            "__symbol__": reference["symbol"],
            "side_name": reference["side_name"],
            "archetype_policy_key": ["long__breakout"],
            "clean_exec": [1.0],
        }
    ).to_parquet(source_path, index=False)
    pd.DataFrame(
        {
            "__ts__": reference["timestamp"],
            "__symbol__": reference["symbol"],
            "side_name": reference["side_name"],
            "__first_touch_mae_to_sl__": [0.35],
            "__first_touch_hit__": [1.0],
            "__first_touch_stop__": [0.0],
            "__first_touch_timeout__": [0.0],
        }
    ).to_parquet(label_root / "train_global_long_5_2026_07.parquet", index=False)
    policy_path.write_text(
        json.dumps({"reference_candidates_path": reference_path.name}),
        encoding="utf-8",
    )

    result = module.materialize(
        policy_path=policy_path,
        matrix_sources=[source_path],
        output_name="reference_mae.parquet",
        label_root=label_root,
    )

    enriched = pd.read_parquet(policy_dir / "reference_mae.parquet")
    assert result["label_mae_join_match_rate"] == 1.0
    assert enriched["first_touch_mae_to_sl"].tolist() == [0.35]
    assert enriched["first_touch_tp_hit"].tolist() == [1.0]
    assert enriched["first_touch_stop"].tolist() == [0.0]
    assert enriched["first_touch_timeout"].tolist() == [0.0]
