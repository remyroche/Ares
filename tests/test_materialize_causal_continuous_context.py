from __future__ import annotations

import json

import numpy as np
import pandas as pd

from extreme_price_movements.causal_market_regime_systems import (
    CONTINUOUS_CONTEXT_FEATURE_KEYS,
    CONTINUOUS_CONTEXT_SOURCE_CONTRACT,
)
from scripts.materialize_causal_continuous_context import materialize


def test_continuous_sidecar_is_candidate_exact_and_never_looks_ahead(tmp_path) -> None:
    source_utc = pd.date_range("2023-01-01", periods=4_900, freq="h", tz="UTC")
    panel = pd.DataFrame({"source_utc": source_utc})
    for number, source in enumerate(CONTINUOUS_CONTEXT_SOURCE_CONTRACT.values(), start=1):
        panel[source] = np.sin(np.arange(len(panel)) / float(number + 3)) + number
    panel_path = tmp_path / "panel.parquet"
    panel.to_parquet(panel_path, index=False)
    candidates = pd.DataFrame({
        "candidate_id": ["a", "b"],
        "__ts__": [source_utc[-20], source_utc[-1]],
        "__symbol__": ["BTC", "ETH"],
        "side_name": ["long", "short"],
    })
    candidate_path = tmp_path / "candidates.parquet"
    candidates.to_parquet(candidate_path, index=False)
    output = materialize(
        panel_path=panel_path,
        output_dir=tmp_path / "output",
        evaluation_start=str(source_utc[-48]),
        candidate_path=candidate_path,
    )
    result = pd.read_parquet(output / "candidate_causal_continuous_context.parquet")
    assert result["candidate_id"].tolist() == ["a", "b"]
    assert result.loc[:, list(CONTINUOUS_CONTEXT_FEATURE_KEYS)].notna().all().all()
    assert (result["source_utc"] <= result["__ts__"]).all()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["contract"]["no_outcomes_or_candidate_scores"] is True
    assert "no GMM" in manifest["contract"]["model_inputs"]
