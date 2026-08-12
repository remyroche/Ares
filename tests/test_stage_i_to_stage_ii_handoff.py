from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pandas as pd

from extreme_price_movements.stage_i_to_stage_ii_handoff import (
    StageIToStageIIHandoffSpec,
    materialize_stage_i_to_stage_ii_handoff,
)


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _setup(tmp_path: Path) -> tuple[Path, Path]:
    start = pd.Timestamp("2025-01-01T00:00:00Z")
    rows: list[dict] = []
    inputs = tmp_path / "inputs"
    for side, offset in (("long", 0), ("short", 4)):
        contract_rows = []
        feature_rows = []
        for number in range(4):
            signal = start + pd.Timedelta(hours=offset + number)
            candidate = f"{side}-{number}"
            rows.append({
                "candidate_id": candidate, "side_name": side,
                "decision_ts": signal + pd.Timedelta(hours=1),
                "label_available_ts": signal + pd.Timedelta(hours=13),
                "exact_gross_bps": 125.0, "exact_net_bps": 25.0,
                "base_direct_score": .2, "base_strict_oof_available": number != 0,
                "base_state_p0": .2, "base_state_p1": .3, "base_state_p2": .5,
            })
            identity = {"candidate_id": candidate, "__ts__": signal, "__symbol__": "BTC"}
            feature_rows.append({**identity, "causal_regime": float(number), "causal_context": 1.0})
            contract_rows.append({**identity, "side_name": side,
                                  "decision_ts": signal + pd.Timedelta(hours=1),
                                  "label_available_ts": signal + pd.Timedelta(hours=13)})
        side_root = inputs / side
        side_root.mkdir(parents=True)
        features = side_root / "features.parquet"
        contract = side_root / "contract.parquet"
        pd.DataFrame(feature_rows).to_parquet(features, index=False)
        pd.DataFrame(contract_rows).to_parquet(contract, index=False)
        _write_json(side_root / "manifest.json", {
            "status": "complete", "side": side,
            "artifact_sha256": {features.name: _sha(features), contract.name: _sha(contract)},
        })
    oos = tmp_path / "oos"
    oos.mkdir()
    predictions = oos / "strict_oof_predictions.parquet"
    pd.DataFrame(rows).to_parquet(predictions, index=False)
    _write_json(oos / "manifest.json", {
        "status": "complete", "files": {predictions.name: _sha(predictions)},
        "shared_population_contract_sha256": "a" * 64,
    })
    return oos, inputs


def test_handoff_keeps_only_strict_base_oof_rows_and_causal_context(tmp_path: Path) -> None:
    oos, inputs = _setup(tmp_path)
    output_dir = tmp_path / "handoff"
    manifest = materialize_stage_i_to_stage_ii_handoff(StageIToStageIIHandoffSpec(oos, inputs, output_dir))
    ledger = pd.read_parquet(output_dir / "direct_stage_i_ledger.parquet")
    assert len(ledger) == 6
    assert ledger.base_strict_oof_available.all()
    assert manifest["source_rows"] == 8
    assert manifest["excluded_non_strict_base_oof_rows"] == 2
    assert manifest["base_state_width"] == 3
    assert {"causal_regime", "causal_context", "base_output_entropy", "symbol", "signal_close_ts"}.issubset(ledger.columns)
    assert ledger.decision_ts.eq(ledger.signal_close_ts + pd.Timedelta(hours=1)).all()
