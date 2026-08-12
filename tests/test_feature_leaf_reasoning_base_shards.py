from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.merge_feature_leaf_reasoning_base_shards import BaseShardMergeError, merge_shards


def _write_shard(root: Path, *, transport: str, arm: str = "F0_current_frozen") -> Path:
    root.mkdir()
    pd.DataFrame([
        {
            "transport": transport, "arm": arm, "scope": "global", "period": None,
            "side_name": None, "top_fraction": 0.10, "net_bps_per_trade": 1.0,
            "feature_contract_99pct_pass": True,
        }
    ]).to_parquet(root / "base_feature_ablation_results.parquet", index=False)
    pd.DataFrame([{"transport": transport, "arm": arm, "gate": "top10"}]).to_parquet(
        root / "base_feature_transport_gates.parquet", index=False
    )
    pd.DataFrame([{"transport": transport, "arm": arm, "feature": "x", "finite_coverage": 1.0}]).to_parquet(
        root / "base_feature_contract_coverage.parquet", index=False
    )
    pd.DataFrame(columns=["transport", "arm", "status"]).to_parquet(
        root / "base_feature_rejected_arms.parquet", index=False
    )
    (root / "base_feature_arm_lineage.json").write_text("[]\n", encoding="utf-8")
    (root / "feature_portability_dispositions.parquet").write_bytes(b"same-stage-a-audit")
    (root / "run_manifest.json").write_text(json.dumps({"source_contract": {"geometry": "TP6/SL4/H12"}}), encoding="utf-8")
    return root


def test_shard_merge_preserves_disjoint_transport_pairs_and_defers_selection(tmp_path: Path) -> None:
    a = _write_shard(tmp_path / "a", transport="transport_a_2023q4_to_2024h1")
    b = _write_shard(tmp_path / "b", transport="transport_b_2024h1_to_2024h2_to_date")
    manifest = merge_shards(destination=tmp_path / "merged", shards=[a, b])
    assert manifest["status"] == "BASE_FEATURE_SHARDS_MERGED_INCOMPLETE"
    table = pd.read_parquet(tmp_path / "merged" / "base_feature_ablation_results.parquet")
    assert set(table["transport"]) == {
        "transport_a_2023q4_to_2024h1", "transport_b_2024h1_to_2024h2_to_date"
    }


def test_shard_merge_rejects_duplicate_transport_arm_pair(tmp_path: Path) -> None:
    a = _write_shard(tmp_path / "a", transport="transport_a_2023q4_to_2024h1")
    b = _write_shard(tmp_path / "b", transport="transport_a_2023q4_to_2024h1")
    with pytest.raises(BaseShardMergeError, match="duplicate immutable"):
        merge_shards(destination=tmp_path / "merged", shards=[a, b])
