from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from extreme_price_movements.inference.causal_sr_c1_lva_bundle import (
    CausalSRC1LVABundle,
    OUTPUT_COLUMNS,
)
from scripts import run_causal_sr_heads as source


class _Regressor:
    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.repeat(0.4, len(frame))


class _Classifier:
    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return np.tile(np.asarray([0.3, 0.7]), (len(frame), 1))


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_c1_lva_bundle_scores_target_free_zone_rows_with_bound_contract(tmp_path: Path) -> None:
    profile = source.PROFILE_CONTEXT_GROUPS["levels"]
    conditional = (*source.CONDITIONAL_FEATURES, *profile, source.PROFILE_CONTEXT_AVAILABLE)
    paths = {
        "prior_model": tmp_path / "prior_model.joblib",
        "conditional_model": tmp_path / "conditional_model.joblib",
        "break_model": tmp_path / "break_model.joblib",
        "magnitude_model": tmp_path / "magnitude_model.joblib",
    }
    for role, path in paths.items():
        joblib.dump(_Classifier() if role == "break_model" else _Regressor(), path)
    manifest = {
        "schema": "causal-sr-c1-lva-inference-bundle-v1",
        "status": "SEALED_NO_ORDER_C1_LVA_SOURCE_BUNDLE",
        "features": {"prior": list(source.PRIOR_FEATURES), "conditional": list(conditional), "profile": list(profile)},
        "files": {role: {"name": path.name, "sha256": _hash(path)} for role, path in paths.items()},
    }
    (tmp_path / "bundle_manifest.json").write_text(json.dumps(manifest))
    rows = []
    for side in ("support", "resistance"):
        row = {column: 0.0 for column in conditional}
        row.update({
            "__symbol__": "A/USD:USD", "snapshot_ts": pd.Timestamp("2026-09-01T00:00:00Z"),
            "target_kind": "entry", "target_id": "a", "candidate_id": "a",
            "zone_side": side, "zone_distance_atr": 1.0,
        })
        rows.append(row)
    scored = CausalSRC1LVABundle.load(tmp_path).score_zone_rows(pd.DataFrame(rows))
    assert len(scored) == 1
    assert scored.sr_snapshot_available.tolist() == [True]
    assert set(OUTPUT_COLUMNS).issubset(scored.columns)
    assert scored.sr_long_support_hold_strength.iloc[0] == 0.4
    assert scored.sr_long_resistance_break_probability.iloc[0] == 0.7
