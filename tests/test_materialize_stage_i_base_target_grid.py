from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.stage_i_base_target_ablation import BaseTargetAblationError, file_sha256
from scripts.materialize_stage_i_base_target_grid import materialize
import scripts.materialize_stage_i_base_target_grid as target_grid


def test_materializer_publishes_exact_60_arm_hash_bound_surface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "paths"
    source.mkdir()
    signal = pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"])
    ledger = pd.DataFrame({
        "candidate_id": ["a", "b"], "__ts__": signal, "__symbol__": ["X", "Y"],
        "side_name": ["long", "short"], "decision_ts": signal + pd.Timedelta(hours=1),
        "entry_ts": signal + pd.Timedelta(hours=1),
        "entry_price": [100., 100.], "atr_1h": [1., 1.], "path_complete": [True, False],
        "label_available_ts": signal + pd.Timedelta(hours=13),
        "path_start_ts": signal + pd.Timedelta(hours=1),
        "path_end_exclusive": signal + pd.Timedelta(hours=13),
        "causal_regime": ["trend", "chop"],
    })
    ledger_path = source / "candidate_paths.parquet"
    ledger.to_parquet(ledger_path, index=False)
    high = np.full((2, 720), 100., dtype=np.float32)
    low = np.full((2, 720), 100., dtype=np.float32)
    close = np.full((2, 720), 100., dtype=np.float32)
    high[0, 0] = 104.1
    paths_path = source / "h12_paths.npz"
    identity_sha = np.vstack([
        np.frombuffer(__import__("hashlib").sha256(
            (str(row.candidate_id) + "\x1f" + pd.Timestamp(row.entry_ts).isoformat()).encode()
        ).digest(), dtype=np.uint8)
        for row in ledger[["candidate_id", "entry_ts"]].itertuples(index=False)
    ])
    np.savez(
        paths_path, high=high, low=low, close=close,
        entry_open=ledger.entry_price.to_numpy(float),
        path_start_ns=pd.to_datetime(ledger.path_start_ts, utc=True).astype("int64").to_numpy(np.int64),
        identity_sha256=identity_sha,
    )
    minute_inventory = {
        "schema": "stage_i_target_minute_source_inventory_v1",
        "minute_root": str((tmp_path / "minute").resolve()), "rows": [],
        "content_hash_policy": "test", "inventory_sha256": "b" * 64,
    }
    minute_inventory_path = source / "minute_source_inventory.json"
    minute_inventory_path.write_text(json.dumps(minute_inventory))
    manifest = {
        "schema": "stage_i_base_target_exact_h12_path_pack_v2", "status": "complete",
        "entry_convention": "signal_timestamp_plus_1h_exact_minute_open",
        "horizon_minutes": 720,
        "materializer_source_fingerprint": {"contract_sha256": "c" * 64},
        "minute_source_inventory": {"path": "minute_source_inventory.json", "inventory_sha256": "b" * 64},
        "causal_regime_contract": {
            "column": "causal_regime", "causal_at_decision_time": True,
            "diagnostic_noncausal": False, "source_manifest_sha256": "a" * 64,
        },
        "artifact_sha256": {
            "candidate_paths.parquet": file_sha256(ledger_path),
            "h12_paths.npz": file_sha256(paths_path),
            "minute_source_inventory.json": file_sha256(minute_inventory_path),
        },
    }
    (source / "manifest.json").write_text(json.dumps(manifest))
    out = tmp_path / "out"
    result = materialize(source, out)
    assert result["geometries"] == 15 and result["target_arms"] == 60
    assert result["path_primitive_reuse"] == {
        "schema": "stage_i_h12_path_primitive_reuse_v1",
        "raw_ohlc_normalisations": 1,
        "distinct_upper_first_touch_traversals": 5,
        "distinct_lower_first_touch_traversals": 3,
        "geometry_contracts_derived": 15,
        "target_neutral": True,
    }
    surface = pd.read_parquet(out / "target_repair_labels.parquet")
    assert len(surface) == 30
    assert surface.loc[surface.candidate_id.eq("b"), "target_valid"].eq(False).all()
    target_columns = [name for name in surface if name == "S_target" or name.startswith("O_")]
    assert target_columns == ["S_target", "O_a0p25_target", "O_a0p33_target", "O_a0p5_target"]
    assert result["artifact_sha256"]["target_repair_labels.parquet"] == file_sha256(out / "target_repair_labels.parquet")
    resumed = materialize(source, out, resume=True)
    assert resumed["request_sha256"] == result["request_sha256"]
    monkeypatch.setattr(
        target_grid, "_materializer_source_fingerprint",
        lambda: {"schema": "test", "contract_sha256": "0" * 64},
    )
    with pytest.raises(BaseTargetAblationError, match="request/source lineage drift"):
        materialize(source, out, resume=True)
    monkeypatch.undo()
    # Even a correctly rehashed path pack must fail when it declares outcomes
    # available before the exclusive H12 endpoint.
    ledger.loc[0, "label_available_ts"] -= pd.Timedelta(minutes=1)
    ledger.to_parquet(ledger_path, index=False)
    manifest["artifact_sha256"]["candidate_paths.parquet"] = file_sha256(ledger_path)
    (source / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(BaseTargetAblationError, match="premature availability"):
        materialize(source, tmp_path / "leaky")
    ledger.loc[0, "label_available_ts"] += pd.Timedelta(minutes=1)
    ledger.loc[0, "entry_price"] += 1.0
    ledger.to_parquet(ledger_path, index=False)
    manifest["artifact_sha256"]["candidate_paths.parquet"] = file_sha256(ledger_path)
    (source / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(BaseTargetAblationError, match="entry_price differs"):
        materialize(source, tmp_path / "wrong_entry")
