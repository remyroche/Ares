from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "materialize_execution_ev_retrospective_geometry",
    ROOT / "scripts" / "materialize_execution_ev_retrospective_geometry.py",
)
assert SPEC and SPEC.loader
materializer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = materializer
SPEC.loader.exec_module(materializer)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _policy(path: Path) -> Path:
    payload = {
        "exit_geometry_contract": {"replay_timeframe": "1m"},
        "strategies": [
            {"selected": True, "side": "long", "exit_geometry_scope": "side_parent"},
            {"selected": True, "side": "short", "exit_geometry_scope": "side_parent"},
            {
                "selected": True,
                "side": "long",
                "exit_geometry_scope": "side_archetype",
                "policy_archetype": "policy_archetype_long__clean",
            },
            {
                "selected": True,
                "side": "short",
                "exit_geometry_scope": "side_archetype",
                "policy_archetype": "policy_archetype_short__clean",
            },
        ],
    }
    path.write_text(json.dumps(payload))
    return path


def _hourly_store(root: Path, *, nonfinite: bool = False) -> None:
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    index = pd.date_range(start, periods=2_900, freq="1h", tz="UTC")
    for number, symbol in enumerate(("AAA/USD:USD", "BBB/USD:USD"), start=1):
        close = 100.0 * number + np.arange(len(index), dtype=float) * 0.02
        frame = pd.DataFrame(
            {
                "ts": index,
                "open": close - 0.1,
                "high": close + 0.2,
                "low": close - 0.2,
                "close": close,
            }
        )
        if nonfinite and symbol == "BBB/USD:USD":
            frame.loc[frame["ts"].eq(pd.Timestamp("2026-04-30T19:00:00Z")), "close"] = np.nan
        directory = root / "ohlcv" / f"symbol={symbol.replace('/', '_')}" / "year=2026"
        directory.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(directory / "bars.parquet", index=False)


def _inputs(tmp_path: Path, *, bad_geometry_availability: bool = False, drop_geometry: bool = False, nonfinite: bool = False) -> dict[str, Path]:
    timestamp = pd.Timestamp("2026-04-30T19:00:00Z")
    population = pd.DataFrame(
        {
            "__ts__": [timestamp, timestamp],
            "__symbol__": ["AAA/USD:USD", "BBB/USD:USD"],
            "side_name": ["long", "short"],
            "candidate_id": ["long-1", "short-1"],
            "execution_decision_utc": [timestamp + pd.Timedelta(hours=1)] * 2,
            "feature_available_at": [timestamp, timestamp],
        }
    )
    population_path = tmp_path / "population.parquet"
    population.to_parquet(population_path, index=False)
    population_manifest = tmp_path / "population.manifest.json"
    population_manifest.write_text(
        json.dumps(
            {
                "schema": "packb_final_refits_forward_v1",
                "output": {"sha256": _sha(population_path)},
            }
        )
    )
    geometry = pd.DataFrame(
        {
            "__ts__": [timestamp, timestamp],
            "__symbol__": ["AAA/USD:USD", "BBB/USD:USD"],
            "side_name": ["long", "short"],
            "candidate_id": ["long-1", "short-1"],
            "__barrier_pct__": [0.01, 0.012],
            "policy_archetype": ["long__clean", "short__clean"],
            "geometry_available_at": [
                timestamp + pd.Timedelta(hours=2) if bad_geometry_availability else timestamp,
                timestamp,
            ],
        }
    )
    if drop_geometry:
        geometry = geometry.iloc[:1]
    geometry_path = tmp_path / "geometry.parquet"
    geometry.to_parquet(geometry_path, index=False)
    geometry_manifest = tmp_path / "geometry.manifest.json"
    geometry_manifest.write_text(
        json.dumps(
            {
                "schema": "execution_ev_frozen_decision_geometry_v1",
                "outcomes_used": False,
                "output": {"sha256": _sha(geometry_path)},
            }
        )
    )
    ohlcv_root = tmp_path / "ohlcv_root"
    _hourly_store(ohlcv_root, nonfinite=nonfinite)
    return {
        "population": population_path,
        "population_manifest": population_manifest,
        "geometry": geometry_path,
        "geometry_manifest": geometry_manifest,
        "policy": _policy(tmp_path / "policy.json"),
        "ohlcv_root": ohlcv_root,
    }


def _run(tmp_path: Path, **kwargs: object) -> dict[str, Path]:
    inputs = _inputs(tmp_path, **kwargs)
    return materializer.materialize(
        population_path=inputs["population"],
        population_manifest_path=inputs["population_manifest"],
        geometry_path=inputs["geometry"],
        geometry_manifest_path=inputs["geometry_manifest"],
        policy_path=inputs["policy"],
        ohlcv_root=inputs["ohlcv_root"],
        output_dir=tmp_path / "output",
    )


def test_materializes_exact_causal_geometry_and_signal_atr(tmp_path: Path) -> None:
    result = _run(tmp_path)
    context = pd.read_parquet(result["policy_context"])
    targets = pd.read_parquet(result["path_targets"])
    manifest = json.loads(result["manifest"].read_text())

    assert len(context) == len(targets) == 2
    assert context["policy_archetype"].tolist() == ["long__clean", "short__clean"]
    assert context["execution_geometry_source"].eq("frozen_local_archetype").all()
    assert targets["__path_auxiliary_atr_fraction__"].gt(0).all()
    assert targets["__path_auxiliary_atr_available_at__"].eq(targets["__ts__"]).all()
    assert manifest["outcomes_used"] is False
    assert manifest["promotion_status"] == "non_promotable_retrospective_only"
    assert manifest["timing"]["label_horizon_hours"] == 12
    assert manifest["atr_contract"]["warmup_days"] == 90


def test_rejects_incomplete_exact_geometry_identity(tmp_path: Path) -> None:
    with pytest.raises(materializer.RetrospectiveGeometryError, match="exact one-to-one"):
        _run(tmp_path, drop_geometry=True)
    assert not (tmp_path / "output" / "policy_context.parquet").exists()


def test_rejects_geometry_available_after_decision(tmp_path: Path) -> None:
    with pytest.raises(materializer.RetrospectiveGeometryError, match="availability occurs after decision"):
        _run(tmp_path, bad_geometry_availability=True)


def test_rejects_nonfinite_signal_time_ohlcv_without_fill(tmp_path: Path) -> None:
    with pytest.raises(materializer.RetrospectiveGeometryError, match="nonfinite or invalid"):
        _run(tmp_path, nonfinite=True)


def test_rejects_geometry_manifest_without_outcome_free_provenance(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    inputs["geometry_manifest"].write_text(
        json.dumps(
            {
                "schema": "execution_ev_frozen_decision_geometry_v1",
                "output": {"sha256": _sha(inputs["geometry"])},
            }
        )
    )
    with pytest.raises(materializer.RetrospectiveGeometryError, match="outcomes_used=false"):
        materializer.materialize(
            population_path=inputs["population"],
            population_manifest_path=inputs["population_manifest"],
            geometry_path=inputs["geometry"],
            geometry_manifest_path=inputs["geometry_manifest"],
            policy_path=inputs["policy"],
            ohlcv_root=inputs["ohlcv_root"],
            output_dir=tmp_path / "output",
        )
