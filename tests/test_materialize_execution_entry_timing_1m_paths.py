from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.execution_entry_timing_meta import (
    EntryTimingTargetSpec,
    _decode_path,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "materialize_execution_entry_timing_1m_paths",
    ROOT / "scripts" / "materialize_execution_entry_timing_1m_paths.py",
)
assert SPEC and SPEC.loader
materializer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = materializer
SPEC.loader.exec_module(materializer)


def _candidate(decision: pd.Timestamp, *, candidate_id: str = "candidate-0") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal": [decision - pd.Timedelta(hours=1)],
            "instrument": ["BTC/USD:USD"],
            "direction": ["long"],
            "id": [candidate_id],
            "decision": [decision],
            "atr": [1.25],
            "fee_return": [0.001],
            "entry_bps": [4.0],
            "exit_bps": [6.0],
        }
    )


def _mapping_args(path: Path, **overrides: object) -> Namespace:
    values: dict[str, object] = {
        "input": path,
        "timestamp_col": "signal",
        "symbol_col": "instrument",
        "side_col": "direction",
        "candidate_id_col": "id",
        "decision_ts_col": "decision",
        "atr_col": "atr",
        "fee_col": "fee_return",
        "entry_spread_col": "entry_bps",
        "exit_spread_col": "exit_bps",
        "decision_delay_minutes": 60,
        "candidate_batch_rows": 1,
    }
    values.update(overrides)
    return Namespace(**values)


def _write_store(data_root: Path, decision: pd.Timestamp, *, gap: int | None = None) -> Path:
    root = data_root / "exchanges" / "krakenfutures" / "execution_1m" / "ohlcv" / "symbol=BTC_USD:USD" / "year=2026"
    root.mkdir(parents=True)
    index = pd.date_range(decision, periods=720, freq="min", tz="UTC")
    if gap is not None:
        index = index.delete(gap)
    pd.DataFrame(
        {"ts": index, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 3.0}
    ).to_parquet(root / "part-1767229200-1767272340.parquet", index=False)
    return data_root


def _target_manifest(path: Path) -> Path:
    payload: dict[str, object] = {
        "schema": "execution_ev_12h_hourly_policy_labels_v2",
        "prediction_role": "execution_ev_12h_labels",
    }
    payload["prediction_role_manifest_sha256"] = materializer._manifest_hash(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _deployed_target_manifest(
    path: Path,
    *,
    economics: str = "current_frozen_spread_counterfactual",
) -> Path:
    payload: dict[str, object] = {
        "schema": "execution_ev_deployed_policy_1m_labels_v1",
        "prediction_role": "execution_ev_12h_labels",
        "timing": {
            "signal_to_decision_minutes": 60,
            "horizon_minutes": 720,
            "label_available_at": "decision + full replay horizon",
        },
        "exit_policy_contract": {"horizon_minutes": 720},
        "store": {
            "contract": "canonical_kraken_execution_1m_immutable_read_only_v1"
        },
        "historical_lineage": {
            "oof_status": "not_oof",
            "execution_parity_claim": False,
            "promotion_eligible": False,
            "economics": economics,
        },
    }
    payload["prediction_role_manifest_sha256"] = materializer._manifest_hash(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _materialize_args(input_path: Path, data_root: Path, target: Path, output: Path, **overrides: object) -> Namespace:
    values = vars(_mapping_args(input_path)).copy()
    values.update(
        {
            "data_root": data_root,
            "execution_ev_target_manifest": target,
            "output": output,
            "manifest": output.with_suffix(".manifest.json"),
            "missing_report": output.with_suffix(".missing.json"),
            "completed_through_utc": "2026-01-03T00:00:00Z",
            "allow_subset": False,
        }
    )
    values.update(overrides)
    return Namespace(**values)


def test_stage_uses_decision_timestamp_and_exact_identity(tmp_path: Path) -> None:
    decision = pd.Timestamp("2026-01-01T01:00:00Z")
    source = tmp_path / "source.parquet"
    _candidate(decision).to_parquet(source, index=False)
    output = tmp_path / "stage.parquet"

    result = materializer.stage(_mapping_args(source, output=output))
    staged = pd.read_parquet(result["staging"])

    assert staged.loc[0, "timestamp"] == decision
    assert staged.loc[0, "symbol"] == "BTC/USD:USD"
    assert staged.loc[0, "candidate_id"] == "candidate-0"


def test_materializes_complete_path_and_signed_manifest(tmp_path: Path) -> None:
    decision = pd.Timestamp("2026-01-01T01:00:00Z")
    source = tmp_path / "source.parquet"
    _candidate(decision).to_parquet(source, index=False)
    data_root = _write_store(tmp_path / "data", decision)
    target = _target_manifest(tmp_path / "target.json")
    output = tmp_path / "paths.parquet"

    result = materializer.materialize(_materialize_args(source, data_root, target, output))
    frame = pd.read_parquet(result["paths"])
    manifest = json.loads(result["manifest"].read_text())
    path = json.loads(frame.loc[0, "execution_future_path"])

    assert frame.loc[0, list(materializer.IDENTITY)].tolist() == [
        decision - pd.Timedelta(hours=1), "BTC/USD:USD", "long", "candidate-0"
    ]
    assert len(path["timestamp"]) == len(path["open"]) == 720
    assert np.asarray(path["open"], dtype=np.float32).dtype == np.float32
    decoded = _decode_path(frame.loc[0, "execution_future_path"], EntryTimingTargetSpec(), row=0)
    assert len(decoded) == 720
    assert decoded["timestamp"].iloc[0] == decision
    assert frame.loc[0, "decision_price"] == pytest.approx(100.0)
    assert frame.loc[0, "atr_1h"] == pytest.approx(1.25)
    assert frame.loc[0, "fee"] == pytest.approx(0.001)
    assert frame.loc[0, "entry_spread"] == pytest.approx(4.0)
    assert frame.loc[0, "exit_spread"] == pytest.approx(6.0)
    assert "cost_return" not in frame.columns
    assert manifest["schema"] == materializer.SCHEMA
    assert manifest["cost_accounting"] == "fee_once_entry_spread_once_exit_spread_once"
    assert manifest["source_artifact_sha256"] == materializer._sha256(output)
    assert manifest["prediction_role_manifest_sha256"] == materializer._manifest_hash(manifest)


def test_derives_absolute_atr_from_decision_price_and_fraction(tmp_path: Path) -> None:
    decision = pd.Timestamp("2026-01-01T01:00:00Z")
    source = tmp_path / "source.parquet"
    candidate = _candidate(decision).drop(columns="atr")
    candidate["atr_fraction"] = 0.0125
    candidate.to_parquet(source, index=False)
    data_root = _write_store(tmp_path / "data", decision)
    target = _target_manifest(tmp_path / "target.json")
    output = tmp_path / "paths.parquet"

    result = materializer.materialize(
        _materialize_args(
            source,
            data_root,
            target,
            output,
            atr_col=None,
            atr_fraction_col="atr_fraction",
        )
    )
    frame = pd.read_parquet(result["paths"])
    manifest = json.loads(result["manifest"].read_text())

    assert frame.loc[0, "decision_price"] == pytest.approx(100.0)
    assert frame.loc[0, "atr_1h"] == pytest.approx(1.25)
    assert manifest["atr"]["input_mode"] == "decision_price_times_atr_fraction"


def test_gap_fails_by_default_and_reports_missing_window(tmp_path: Path) -> None:
    decision = pd.Timestamp("2026-01-01T01:00:00Z")
    source = tmp_path / "source.parquet"
    _candidate(decision).to_parquet(source, index=False)
    data_root = _write_store(tmp_path / "data", decision, gap=120)
    target = _target_manifest(tmp_path / "target.json")
    output = tmp_path / "paths.parquet"
    args = _materialize_args(source, data_root, target, output)

    with pytest.raises(ValueError, match="lack an exact completed"):
        materializer.materialize(args)

    report = json.loads(args.missing_report.read_text())
    assert report["incomplete_rows"] == 1
    assert report["missing_windows"][0]["reason"] == "missing_or_nonfinite_minutes=1"
    assert not output.exists()


def test_subset_is_explicit_and_preserves_cost_decomposition(tmp_path: Path) -> None:
    decision = pd.Timestamp("2026-01-01T01:00:00Z")
    source = tmp_path / "source.parquet"
    rows = pd.concat([_candidate(decision), _candidate(decision + pd.Timedelta(hours=13), candidate_id="missing")], ignore_index=True)
    rows.to_parquet(source, index=False)
    data_root = _write_store(tmp_path / "data", decision)
    target = _target_manifest(tmp_path / "target.json")
    output = tmp_path / "paths.parquet"
    result = materializer.materialize(_materialize_args(source, data_root, target, output, allow_subset=True))

    frame = pd.read_parquet(result["paths"])
    manifest = json.loads(result["manifest"].read_text())
    assert frame["candidate_id"].tolist() == ["candidate-0"]
    assert manifest["rows"] == {"requested": 2, "output": 1, "subset": True}
    assert manifest["cost_columns"]["policy"].startswith("values are carried without deduction")


def test_rejects_duplicate_identity_and_bad_target_signature(tmp_path: Path) -> None:
    decision = pd.Timestamp("2026-01-01T01:00:00Z")
    source = tmp_path / "source.parquet"
    pd.concat([_candidate(decision), _candidate(decision)]).to_parquet(source, index=False)
    with pytest.raises(ValueError, match="duplicate exact identity"):
        materializer.stage(_mapping_args(source, output=tmp_path / "stage.parquet"))

    target = tmp_path / "bad-target.json"
    target.write_text(json.dumps({"schema": "execution_ev_12h_hourly_policy_labels_v2", "prediction_role": "execution_ev_12h_labels", "prediction_role_manifest_sha256": hashlib.sha256(b"wrong").hexdigest()}), encoding="utf-8")
    with pytest.raises(ValueError, match="signature does not verify"):
        materializer._manifest_target(target)


@pytest.mark.parametrize(
    "economics",
    [
        "current_frozen_spread_counterfactual",
        "inverse_quote_notional_current_spread_counterfactual",
    ],
)
def test_accepts_signed_historical_deployed_policy_manifest(
    tmp_path: Path,
    economics: str,
) -> None:
    target = _deployed_target_manifest(
        tmp_path / "deployed-target.json", economics=economics
    )
    digest, signed = materializer._manifest_target(target)
    assert digest == materializer._sha256(target)
    assert signed


def test_rejects_deployed_policy_manifest_without_historical_lineage(
    tmp_path: Path,
) -> None:
    target = _deployed_target_manifest(tmp_path / "deployed-target.json")
    payload = json.loads(target.read_text())
    payload["historical_lineage"] = None
    payload["prediction_role_manifest_sha256"] = materializer._manifest_hash(payload)
    target.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="allowed counterfactual lineage"):
        materializer._manifest_target(target)
