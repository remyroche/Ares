from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "materialize_execution_entry_timing_candidates",
    ROOT / "scripts" / "materialize_execution_entry_timing_candidates.py",
)
assert SPEC and SPEC.loader
materializer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = materializer
SPEC.loader.exec_module(materializer)


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    signal = pd.Timestamp("2026-01-01T00:00:00Z")
    identity = {
        "__ts__": [signal],
        "__symbol__": ["BTC/USD:USD"],
        "side_name": ["long"],
        "candidate_id": ["candidate-0"],
    }
    candidates = tmp_path / "candidates.parquet"
    pd.DataFrame(
        {
            **identity,
            "__path_auxiliary_atr_fraction__": [0.0125],
        }
    ).to_parquet(candidates, index=False)
    labels = tmp_path / "labels.parquet"
    pd.DataFrame(
        {
            **identity,
            "__decision_ts__": [signal + pd.Timedelta(hours=1)],
            "execution_label_end_utc": [signal + pd.Timedelta(hours=13)],
            "execution_fee_return": [0.003],
            "execution_spread_return": [0.001],
            "execution_cost_return": [0.004],
        }
    ).to_parquet(labels, index=False)
    manifest = {
        "schema": "execution_ev_12h_hourly_policy_labels_v2",
        "prediction_role": "execution_ev_12h_labels",
        "source_artifact_sha256": materializer._sha256(labels),
        "source": {
            "candidates": str(candidates),
            "sha256": materializer._sha256(candidates),
        },
        "timing": {
            "signal_timestamp": "__ts__",
            "first_path_timestamp": "__decision_ts__",
            "decision_delay_hours": 1,
            "horizon_hours": 12,
        },
        "accounting": {"cost_contract": "explicit_fee_plus_full_p90_spread"},
    }
    manifest["prediction_role_manifest_sha256"] = materializer._manifest_hash(
        manifest
    )
    target = tmp_path / "target.json"
    target.write_text(json.dumps(manifest), encoding="utf-8")
    return candidates, labels, target


def test_materializes_exact_atr_and_cost_contract(tmp_path: Path) -> None:
    candidates, labels, target = _inputs(tmp_path)
    output = tmp_path / "timing_candidates.parquet"
    result = materializer.materialize(
        Namespace(
            candidates=candidates,
            execution_ev_labels=labels,
            execution_ev_target_manifest=target,
            atr_fraction_col="__path_auxiliary_atr_fraction__",
            output=output,
            manifest=None,
        )
    )
    frame = pd.read_parquet(result["candidates"])
    manifest = json.loads(result["manifest"].read_text(encoding="utf-8"))

    assert frame.loc[0, "atr_fraction"] == pytest.approx(0.0125)
    assert frame.loc[0, "fee"] == pytest.approx(0.003)
    assert frame.loc[0, "entry_spread"] == pytest.approx(5.0)
    assert frame.loc[0, "exit_spread"] == pytest.approx(5.0)
    assert manifest["source_artifact_sha256"] == materializer._sha256(output)
    assert (
        manifest["prediction_role_manifest_sha256"]
        == materializer._manifest_hash(manifest)
    )


def test_rejects_target_hash_mismatch(tmp_path: Path) -> None:
    candidates, labels, target = _inputs(tmp_path)
    frame = pd.read_parquet(candidates)
    frame.loc[0, "__path_auxiliary_atr_fraction__"] = 0.02
    frame.to_parquet(candidates, index=False)

    with pytest.raises(ValueError, match="candidate artifact hash"):
        materializer.materialize(
            Namespace(
                candidates=candidates,
                execution_ev_labels=labels,
                execution_ev_target_manifest=target,
                atr_fraction_col="__path_auxiliary_atr_fraction__",
                output=tmp_path / "out.parquet",
                manifest=None,
            )
        )


@pytest.mark.parametrize(
    "economics",
    [
        "current_frozen_spread_counterfactual",
        "inverse_quote_notional_current_spread_counterfactual",
    ],
)
def test_materializes_historical_deployed_policy_cost_contract(
    tmp_path: Path,
    economics: str,
) -> None:
    signal = pd.Timestamp("2024-01-01T00:00:00Z")
    identity = {
        "__ts__": [signal],
        "__symbol__": ["BTC/USD:USD"],
        "side_name": ["long"],
        "candidate_id": ["historical-0"],
    }
    candidates = tmp_path / "path_targets.parquet"
    pd.DataFrame(
        {**identity, "__path_auxiliary_atr_fraction__": [0.0125]}
    ).to_parquet(candidates, index=False)
    labels = tmp_path / "policy_labels.parquet"
    pd.DataFrame(
        {
            **identity,
            "execution_decision_utc": [signal + pd.Timedelta(hours=1)],
            "execution_label_end_utc": [signal + pd.Timedelta(hours=13)],
            "execution_cost_return": [0.002],
            "execution_entry_half_spread_bps": [4.0],
            "execution_exit_half_spread_bps": [6.0],
        }
    ).to_parquet(labels, index=False)
    target_payload = {
        "schema": "execution_ev_deployed_policy_1m_labels_v1",
        "prediction_role": "execution_ev_12h_labels",
        "source_artifact_sha256": materializer._sha256(labels),
        "source": {
            "path_targets_sha256": materializer._sha256(candidates),
        },
        "timing": {
            "signal_to_decision_minutes": 60,
            "horizon_minutes": 720,
            "label_available_at": "decision + full replay horizon",
        },
        "historical_lineage": {
            "oof_status": "not_oof",
            "execution_parity_claim": False,
            "promotion_eligible": False,
            "economics": economics,
        },
    }
    target_payload["prediction_role_manifest_sha256"] = (
        materializer._manifest_hash(target_payload)
    )
    target = tmp_path / "policy_labels.manifest.json"
    target.write_text(json.dumps(target_payload))
    output = tmp_path / "timing_candidates.parquet"
    result = materializer.materialize(
        Namespace(
            candidates=candidates,
            execution_ev_labels=labels,
            execution_ev_target_manifest=target,
            atr_fraction_col="__path_auxiliary_atr_fraction__",
            output=output,
            manifest=None,
            universe=None,
        )
    )
    frame = pd.read_parquet(result["candidates"])
    manifest = json.loads(result["manifest"].read_text())
    assert frame.loc[0, "fee"] == pytest.approx(0.002)
    assert frame.loc[0, "entry_spread"] == pytest.approx(4.0)
    assert frame.loc[0, "exit_spread"] == pytest.approx(6.0)
    assert "embedded" in manifest["cost_accounting"]["recomposition"]
