from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.materialize_packb_tp6_sl4_h12_labels import _label_candidates_with_minute
from scripts.materialize_stage_i_base_target_path_pack import _regime, materialize
from extreme_price_movements.stage_i_base_target_ablation import file_sha256


def test_missing_causal_regime_inputs_are_explicit_unknown_without_imputation() -> None:
    frame = pd.DataFrame({
        "is_low_vol_regime": [1.0, np.nan],
        "is_high_vol_regime": [0.0, np.nan],
        "is_trending": [1.0, np.nan],
    })
    assert _regime(frame).tolist() == ["low_vol__trend", "causal_unknown"]


def test_path_pack_matches_canonical_signal_plus_1h_exact_minute_open(
    tmp_path: Path, monkeypatch,
) -> None:
    selector = tmp_path / "selector"; selector.mkdir()
    signal = pd.Timestamp("2026-01-02T00:00:00Z")
    ledger = pd.DataFrame({
        "candidate_id": ["x"], "__ts__": [signal], "__symbol__": ["X/USD"],
        "side_name": ["long"], "decision_ts": [signal + pd.Timedelta(hours=1)],
    })
    features = ledger[["candidate_id", "__ts__", "__symbol__"]].copy()
    features["is_low_vol_regime"] = 0.; features["is_high_vol_regime"] = 1.
    features["is_trending"] = 1.
    ledger.to_parquet(selector / "selector_ledger.parquet", index=False)
    features.to_parquet(selector / "selector_features.parquet", index=False)
    (selector / "manifest.json").write_text(json.dumps({
        "status": "complete", "artifact_integrity": {
            "schema": "stage_i_selector_artifact_integrity_v1",
            "selector_ledger_sha256": file_sha256(selector / "selector_ledger.parquet"),
            "selector_features_sha256": file_sha256(selector / "selector_features.parquet"),
        },
    }))
    (selector / "selector_feature_contract.json").write_text(json.dumps({
        "max_feature_columns": 0, "feature_columns": [
            "is_low_vol_regime", "is_high_vol_regime", "is_trending",
        ],
    }))
    index = pd.date_range(signal - pd.Timedelta(hours=15), signal + pd.Timedelta(hours=14), freq="min", inclusive="left")
    base = 100 + np.arange(len(index)) * .001
    minute = pd.DataFrame({
        "open": base, "high": base + .4, "low": base - .4, "close": base + .1,
    }, index=index)
    monkeypatch.setattr(
        "scripts.materialize_stage_i_base_target_path_pack._minute_path_pruned",
        lambda *_args, **_kwargs: minute,
    )
    out = tmp_path / "path_pack"
    result = materialize(selector, tmp_path / "unused", out)
    candidate = pd.read_parquet(out / "candidate_paths.parquet")
    archive = np.load(out / "h12_paths.npz", allow_pickle=False)
    decision = signal + pd.Timedelta(hours=1)
    assert pd.Timestamp(candidate.entry_ts.iloc[0]) == decision
    assert archive["high"][0, 0] == pytest.approx(minute.loc[decision, "high"], abs=1e-5)
    canonical = _label_candidates_with_minute(
        ledger[["candidate_id", "__ts__", "__symbol__", "side_name"]], minute
    )
    assert candidate.entry_price.iloc[0] == canonical.tp6_sl4_entry_price.iloc[0]
    assert candidate.atr_1h.iloc[0] == canonical.atr_1h.iloc[0]
    assert result["entry_convention"] == "signal_timestamp_plus_1h_exact_minute_open"
    assert result["causal_regime_contract"]["diagnostic_noncausal"] is False
    resumed = materialize(selector, tmp_path / "unused", out, resume=True)
    assert resumed["request_sha256"] == result["request_sha256"]
    monkeypatch.setattr(
        "scripts.materialize_stage_i_base_target_path_pack._materializer_source_fingerprint",
        lambda: {"schema": "test", "contract_sha256": "0" * 64},
    )
    with pytest.raises(ValueError, match="request/source lineage drift"):
        materialize(selector, tmp_path / "unused", out, resume=True)


def test_resume_rejects_minute_inventory_content_missing_and_added_drift(
    tmp_path: Path, monkeypatch,
) -> None:
    selector = tmp_path / "selector"; selector.mkdir()
    signal = pd.Timestamp("2026-01-02T00:00:00Z")
    ledger = pd.DataFrame({
        "candidate_id": ["x"], "__ts__": [signal], "__symbol__": ["X/USD"],
        "side_name": ["long"], "decision_ts": [signal + pd.Timedelta(hours=1)],
    })
    features = ledger[["candidate_id", "__ts__", "__symbol__"]].copy()
    features["is_low_vol_regime"] = 0.
    features["is_high_vol_regime"] = 1.
    features["is_trending"] = 1.
    ledger.to_parquet(selector / "selector_ledger.parquet", index=False)
    features.to_parquet(selector / "selector_features.parquet", index=False)
    (selector / "manifest.json").write_text(json.dumps({
        "status": "complete", "artifact_integrity": {
            "schema": "stage_i_selector_artifact_integrity_v1",
            "selector_ledger_sha256": file_sha256(selector / "selector_ledger.parquet"),
            "selector_features_sha256": file_sha256(selector / "selector_features.parquet"),
        },
    }))
    (selector / "selector_feature_contract.json").write_text(json.dumps({
        "max_feature_columns": 0,
        "feature_columns": ["is_low_vol_regime", "is_high_vol_regime", "is_trending"],
    }))
    index = pd.date_range(
        signal - pd.Timedelta(hours=15), signal + pd.Timedelta(hours=14),
        freq="min", inclusive="left",
    )
    base = 100 + np.arange(len(index)) * .001
    minute = pd.DataFrame({
        "open": base, "high": base + .4, "low": base - .4, "close": base + .1,
    }, index=index)
    monkeypatch.setattr(
        "scripts.materialize_stage_i_base_target_path_pack._minute_path_pruned",
        lambda *_args, **_kwargs: minute,
    )
    minute_root = tmp_path / "minute"
    fragment_dir = minute_root / "symbol=X_USD" / "year=2026"
    fragment_dir.mkdir(parents=True)
    fragment = fragment_dir / "legacy.parquet"
    fragment.write_bytes(b"first")
    out = tmp_path / "pack"
    materialize(selector, minute_root, out)

    fragment.write_bytes(b"changed")
    with pytest.raises(ValueError, match="request/source lineage drift"):
        materialize(selector, minute_root, out, resume=True)

    fragment.write_bytes(b"first")
    fragment.unlink()
    with pytest.raises(ValueError, match="request/source lineage drift"):
        materialize(selector, minute_root, out, resume=True)

    fragment.write_bytes(b"first")
    added = fragment_dir / "another-legacy.parquet"
    added.write_bytes(b"second")
    with pytest.raises(ValueError, match="request/source lineage drift"):
        materialize(selector, minute_root, out, resume=True)
