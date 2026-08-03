"""Strict OOF/prequential guards for the candidate-level Stage-C F7 adapter."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from extreme_price_movements.stage_c_prequential_regime_sidecar import (
    F7_FIELDS,
    Inputs,
    PrequentialSidecarError,
    REGIME_SOURCE_FIELDS,
    _prequential_candidate_sidecar,
)


def _inputs(tmp_path, *, invalid_claim: bool = False) -> Inputs:
    source = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    candidates = pd.DataFrame({
        "candidate_id": ["a", "b", "c", "d"], "source_symbol": ["A"] * 4, "side": ["long", "short", "long", "short"],
        "feature_cutoff_ts": source, "decision_ts": source + pd.Timedelta(hours=1),
        "retain_h0_given_clear__valid": [True] * 4,
    })
    candidates_path = tmp_path / "candidates.parquet"; candidates.to_parquet(candidates_path, index=False)
    common = {"source_utc": source, "bocpd_regime_available": [True] * 4, "provenance_partition_bocpd": ["blocked_oof_2022_2025"] * 4,
              "train_end_exclusive_utc_bocpd": source - pd.Timedelta(hours=1), "fit_label_resolution_max_utc_bocpd": source - pd.Timedelta(hours=1)}
    regime = pd.DataFrame({**common, **{name: [0.2] * 4 for name in REGIME_SOURCE_FIELDS}})
    regime_path = tmp_path / "regime.parquet"; regime.to_parquet(regime_path, index=False)
    transition = pd.DataFrame({
        "source_utc": source, "lgbm_transition_available": [True] * 4, "provenance_partition_lgbm": ["blocked_oof_2022_2025"] * 4,
        "train_end_exclusive_utc_lgbm": source + pd.Timedelta(hours=1 if invalid_claim else -1),
        "fit_label_resolution_max_utc_lgbm": source - pd.Timedelta(hours=1),
        **{name: [0.3] * 4 for name in set(F7_FIELDS.values()).difference(REGIME_SOURCE_FIELDS)},
    })
    transition_path = tmp_path / "transition.parquet"; transition.to_parquet(transition_path, index=False)
    manifest = tmp_path / "manifest.json"; manifest.write_text(json.dumps({"schema": "authoritative_soft_regime_transition_sidecars_v1", "status": "SEALED_CAUSAL_SOFT_REGIME_TRANSITION_SIDECARS"}))
    return Inputs(candidates_path, regime_path, transition_path, manifest)


def test_candidate_adapter_writes_only_exact_hourly_strict_prequential_values(tmp_path):
    sidecar, coverage, readiness = _prequential_candidate_sidecar(inputs=_inputs(tmp_path))
    assert sidecar.f7_prequential_valid.all()
    assert sidecar.f7_available_ts.le(sidecar.decision_ts).all()
    assert sidecar.loc[:, list(F7_FIELDS)].notna().all().all()
    assert set(sidecar.f7_provenance_partition) == {"blocked_oof_2022_2025"}
    assert not coverage.empty
    assert readiness["exact_hour_cutoff_join"]


def test_candidate_adapter_rejects_an_available_in_sample_source_row(tmp_path):
    with pytest.raises(PrequentialSidecarError, match="lacks strict prequential provenance"):
        _prequential_candidate_sidecar(inputs=_inputs(tmp_path, invalid_claim=True))
