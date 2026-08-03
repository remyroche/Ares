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
    "materialize_execution_ev_frozen_decision_geometry",
    ROOT / "scripts" / "materialize_execution_ev_frozen_decision_geometry.py",
)
assert SPEC and SPEC.loader
geometry = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = geometry
SPEC.loader.exec_module(geometry)

RETRO_SPEC = importlib.util.spec_from_file_location(
    "materialize_execution_ev_retrospective_geometry",
    ROOT / "scripts" / "materialize_execution_ev_retrospective_geometry.py",
)
assert RETRO_SPEC and RETRO_SPEC.loader
retrospective = importlib.util.module_from_spec(RETRO_SPEC)
sys.modules[RETRO_SPEC.name] = retrospective
RETRO_SPEC.loader.exec_module(retrospective)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_store(
    root: Path,
    *,
    missing_previous_hour: bool = False,
    missing_historical_hour: bool = False,
) -> None:
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    index = pd.date_range(start, periods=3_000, freq="1h", tz="UTC")
    for multiplier, symbol in enumerate(("AAA/USD:USD", "BBB/USD:USD"), start=1):
        close = multiplier * 100.0 + np.arange(len(index), dtype=float) * 0.02
        frame = pd.DataFrame(
            {
                "ts": index,
                "open": close - 0.02,
                "high": close + 0.3,
                "low": close - 0.2,
                "close": close,
            }
        )
        if missing_previous_hour and symbol == "AAA/USD:USD":
            frame = frame.loc[
                ~frame["ts"].eq(pd.Timestamp("2026-04-30T18:00:00Z"))
            ].copy()
        if missing_historical_hour and symbol == "AAA/USD:USD":
            frame = frame.loc[
                ~frame["ts"].eq(pd.Timestamp("2026-02-01T00:00:00Z"))
            ].copy()
        directory = root / "ohlcv" / f"symbol={symbol.replace('/', '_')}" / "year=2026"
        directory.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(directory / "bars.parquet", index=False)


def _policy(path: Path, *, valid_parents: bool = True) -> Path:
    long_id = "long__parent" if valid_parents else "long_not_parent"
    payload = {
        "exit_geometry_contract": {"replay_timeframe": "1m", "horizon_minutes": 1440},
        "strategies": [
            {
                "selected": True,
                "side": "long",
                "exit_geometry_scope": "side_parent",
                "canonical_strategy_id": long_id,
                "strategy_id": long_id,
            },
            {
                "selected": True,
                "side": "short",
                "exit_geometry_scope": "side_parent",
                "canonical_strategy_id": "short__parent",
                "strategy_id": "short__parent",
            },
            {
                "selected": True,
                "side": "long",
                "exit_geometry_scope": "side_archetype",
                "policy_archetype": "policy_archetype_long__fixture",
            },
            {
                "selected": True,
                "side": "short",
                "exit_geometry_scope": "side_archetype",
                "policy_archetype": "policy_archetype_short__fixture",
            },
        ],
    }
    path.write_text(json.dumps(payload))
    return path


def _context(tmp_path: Path) -> tuple[Path, Path, pd.Timestamp]:
    timestamp = pd.Timestamp("2026-04-30T19:00:00Z")
    frame = pd.DataFrame(
        {
            "__ts__": [timestamp, timestamp],
            "__symbol__": ["AAA/USD:USD", "BBB/USD:USD"],
            "side_name": ["long", "short"],
            "candidate_id": ["long-1", "short-1"],
            "selected_top40": [True, True],
            "prediction_source": ["frozen_final_refit", "frozen_final_refit"],
            "execution_decision_utc": [timestamp + pd.Timedelta(hours=1)] * 2,
            "feature_available_at": [timestamp, timestamp],
        }
    )
    context = tmp_path / "packb_forward_context.parquet"
    frame.to_parquet(context, index=False)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "packb_final_refits_forward_v1",
                "status": "frozen_final_refit_preentry_context_not_oos_metrics",
                "contract": {"outcomes_used": False},
                "output": {"sha256": _sha(context)},
            }
        )
    )
    return context, manifest, timestamp


def _inputs(tmp_path: Path, **kwargs: object) -> dict[str, object]:
    context, manifest, timestamp = _context(tmp_path)
    store = tmp_path / "ohlcv"
    _write_store(
        store,
        missing_previous_hour=bool(kwargs.get("missing_previous_hour", False)),
        missing_historical_hour=bool(kwargs.get("missing_historical_hour", False)),
    )
    return {
        "context": context,
        "manifest": manifest,
        "policy": _policy(tmp_path / "policy.json", valid_parents=bool(kwargs.get("valid_parents", True))),
        "store": store,
        "timestamp": timestamp,
    }


def _run(tmp_path: Path, **kwargs: object) -> tuple[dict[str, Path], dict[str, object]]:
    inputs = _inputs(tmp_path, **kwargs)
    result = geometry.materialize(
        context_path=inputs["context"],
        context_manifest_path=inputs["manifest"],
        policy_path=inputs["policy"],
        ohlcv_root=inputs["store"],
        output_dir=tmp_path / "geometry",
    )
    return result, inputs


def test_materializes_exact_lagged_atr_side_parent_geometry_and_is_consumer_compatible(tmp_path: Path) -> None:
    result, inputs = _run(tmp_path)
    frame = pd.read_parquet(result["geometry"])
    manifest = json.loads(result["manifest"].read_text())

    assert frame["policy_archetype"].eq("side_parent").all()
    assert frame["canonical_parent_strategy_id"].tolist() == ["long__parent", "short__parent"]
    assert frame["geometry_available_at"].eq(frame["__ts__"]).all()
    assert np.array_equal(
        frame["__barrier_pct__"].to_numpy(np.float32),
        np.maximum(frame["__lagged_signal_atr_fraction__"].to_numpy(np.float32), np.float32(0.005)),
    )
    assert not np.array_equal(
        frame["__signal_atr_fraction__"].to_numpy(np.float32),
        frame["__lagged_signal_atr_fraction__"].to_numpy(np.float32),
    )
    assert manifest["schema"] == "execution_ev_frozen_decision_geometry_v1"
    assert manifest["outcomes_used"] is False
    assert manifest["policy_parents"]["long"]["canonical_strategy_id"] == "long__parent"

    consumer = retrospective.materialize(
        population_path=inputs["context"],
        population_manifest_path=inputs["manifest"],
        geometry_path=result["geometry"],
        geometry_manifest_path=result["manifest"],
        policy_path=inputs["policy"],
        ohlcv_root=inputs["store"],
        output_dir=tmp_path / "consumer",
    )
    targets = pd.read_parquet(consumer["path_targets"])
    assert np.array_equal(
        targets["__barrier_pct__"].to_numpy(np.float32),
        frame["__barrier_pct__"].to_numpy(np.float32),
    )


def test_rejects_missing_prior_hour_instead_of_asof_or_fill(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path, missing_previous_hour=True)
    with pytest.raises(geometry.FrozenDecisionGeometryError, match="must exist exactly"):
        geometry.materialize(
            context_path=inputs["context"],
            context_manifest_path=inputs["manifest"],
            policy_path=inputs["policy"],
            ohlcv_root=inputs["store"],
            output_dir=tmp_path / "geometry",
        )


def test_retains_audited_historical_gap_without_filling_or_asof(tmp_path: Path) -> None:
    result, _ = _run(tmp_path, missing_historical_hour=True)
    manifest = json.loads(result["manifest"].read_text())
    algo = manifest["inputs"]["ohlcv"]["coverage_by_symbol"]["AAA/USD:USD"]
    assert algo["historical_gap_rows"] == 1
    assert algo["historical_gap_first"] == "2026-02-01T00:00:00+00:00"
    assert manifest["geometry_contract"]["historical_gap_handling"].startswith("retain source gaps")


def test_rejects_noncanonical_side_parent_policy(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path, valid_parents=False)
    with pytest.raises(geometry.FrozenDecisionGeometryError, match="canonical long__parent"):
        geometry.materialize(
            context_path=inputs["context"],
            context_manifest_path=inputs["manifest"],
            policy_path=inputs["policy"],
            ohlcv_root=inputs["store"],
            output_dir=tmp_path / "geometry",
        )


def test_historical_lagged_barrier_parity_when_archive_is_available() -> None:
    """A real archived July row verifies the literal t-1, not t, rule.

    The repository's narrow test checkout may omit market data; a full Ares
    archive runs this directly against the immutable historical label ledger.
    """

    labels = ROOT / "data_perp/artifacts/path_archetype_labels_july20_20260726_v1/path_archetype_labels.parquet"
    if not labels.is_file():
        pytest.skip("historical Ares archive is not present in this checkout")
    result = geometry.historical_lagged_barrier_parity(
        labels_path=labels,
    )
    assert result["rows"] == 177_394
    assert result["mismatches"] == 0
