from __future__ import annotations

import json

import pandas as pd
import pytest

from extreme_price_movements.leaf_reasoning_meta_funnel import (
    MetaTransportGateConfig,
    compare_successor_meta_generations,
)
from scripts.compare_leaf_reasoning_meta_generations import (
    _read_generation,
    _sha256,
    write_immutable_generation_comparison,
)


def _metrics(arm: str, *, uplift: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for transport in ("transport_a", "transport_b"):
        for tail in (0.05, 0.10):
            net = 10.0 + uplift + (1.0 if transport == "transport_b" else 0.0)
            rows.append({
                "arm": arm, "transport_id": transport, "top_fraction": tail,
                "gross_bps": net + 100.0, "cost_bps": 100.0, "net_bps": net,
            })
    return pd.DataFrame(rows)


def _immutable_run(tmp_path, generation: str, arm: str, *, uplift: float) -> tuple[object, pd.DataFrame]:
    root = tmp_path / generation.lower()
    root.mkdir()
    table = _metrics(arm, uplift=uplift)
    metrics = root / "metrics.parquet"
    table.to_parquet(metrics, index=False, compression="zstd")
    manifest = {
        "immutable_output": True,
        "artifact_state": "COMPLETE",
        "table_format": "parquet_zstd",
        "successor": generation,
        "selection_status": "DEVELOPMENT_METRICS_ONLY; final untouched OOS remains required",
        "sha256": {metrics.name: _sha256(metrics)},
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root, table


def test_parquet_generation_comparison_is_hash_bound_and_atomic(tmp_path) -> None:
    s0_root, _ = _immutable_run(tmp_path, "S0", "L0", uplift=0.0)
    s1_root, _ = _immutable_run(tmp_path, "S1", "L4", uplift=1.0)
    s2_root, _ = _immutable_run(tmp_path, "S2", "L4", uplift=2.0)
    sources = {
        "S0": _read_generation(s0_root, "S0"),
        "S1": _read_generation(s1_root, "S1"),
        "S2": _read_generation(s2_root, "S2"),
    }
    table = compare_successor_meta_generations(
        {name: source.metrics for name, source in sources.items()},
        selected_arm_by_generation={"S0": "L0", "S1": "L4", "S2": "L4"},
        gate_config=MetaTransportGateConfig(required_transport_count=2),
    )
    output = write_immutable_generation_comparison(
        table, output_dir=tmp_path / "comparison", sources=sources,
        selected_arms={"S0": "L0", "S1": "L4", "S2": "L4"}, required_transport_count=2,
    )
    assert (output / "meta_generation_comparison.parquet").is_file()
    assert not (output / "meta_generation_comparison.csv").exists()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["artifact_state"] == "COMPLETE"
    assert manifest["terminal_decision"] == "PREDECESSOR_META_REASONING_ADDS_VALUE"
    assert set(manifest["sources"]) == {"S0", "S1", "S2"}
    assert manifest["sha256"]["meta_generation_comparison.parquet"] == _sha256(
        output / "meta_generation_comparison.parquet"
    )
    assert (output / "manifest.sha256").is_file()


def test_generation_loader_rejects_tampered_parquet_and_implicit_csv(tmp_path) -> None:
    root, _ = _immutable_run(tmp_path, "S0", "L0", uplift=0.0)
    (root / "metrics.parquet").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        _read_generation(root, "S0")

    legacy = tmp_path / "legacy"
    legacy.mkdir()
    metrics = legacy / "metrics.csv"
    _metrics("L0", uplift=0.0).to_csv(metrics, index=False)
    (legacy / "manifest.json").write_text(json.dumps({
        "immutable_output": True, "successor": "S0",
        "sha256": {metrics.name: _sha256(metrics)},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="Parquet"):
        _read_generation(legacy, "S0")
    assert len(_read_generation(legacy, "S0", allow_legacy_csv=True).metrics) == 4
