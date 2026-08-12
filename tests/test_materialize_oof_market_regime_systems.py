from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from extreme_price_movements.causal_market_regime_systems import PRIMARY_SEMANTIC_STATE_NAMES
from scripts.materialize_oof_market_regime_systems import (
    compact_observable_columns,
    materialize,
)


def _hourly_panel(rows: int = 1_100) -> pd.DataFrame:
    index = np.arange(rows, dtype=np.float32)
    frame = pd.DataFrame({"source_utc": pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC")})
    for number, name in enumerate(
        (
            "mv__breakout_efficiency_4h__delta_6h",
            "mv__breadth_dispersion__delta_6h",
            "mv__trend_strength__robust_z_6h",
            "mv__atr_level__delta_12h",
            "mv__breadth__delta_6h",
            "mv__dependence__eig1_share_6h",
            "mv__funding_rate__robust_z_6h",
            "mv__oi_value__delta_6h",
            "mv__liquidity__spread__stress_6h",
            "mv__liquidity__volume__stress_12h",
        )
    ):
        frame[name] = np.sin(index / (7.0 + number)) + 0.1 * np.cos(index / (13.0 + number))
    return frame


def test_materializer_emits_five_state_phase_sidecar_and_exact_candidate_join(tmp_path: Path) -> None:
    hourly = _hourly_panel()
    panel_path = tmp_path / "hourly.parquet"
    hourly.to_parquet(panel_path, index=False)
    start = pd.Timestamp("2025-02-01T00:00:00Z")
    timestamps = hourly.loc[hourly["source_utc"].ge(start), "source_utc"].iloc[::3].reset_index(drop=True)
    candidates = pd.DataFrame(
        {
            "candidate_id": [f"candidate_{index}" for index in range(len(timestamps))],
            "__ts__": timestamps,
            "__symbol__": "BTC/USD:USD",
            "side_name": "long",
        }
    )
    candidate_path = tmp_path / "candidates.parquet"
    candidates.to_parquet(candidate_path, index=False)

    output = materialize(
        panel_path=panel_path,
        output_dir=tmp_path / "output",
        evaluation_start=start.isoformat(),
        evaluation_end="2025-02-15T00:00:00Z",
        candidate_path=candidate_path,
        frequency="month",
        max_features_per_view=8,
        max_lag_hours=1,
    )
    timeline = pd.read_parquet(output / "hourly_oof_market_regimes.parquet")
    candidate = pd.read_parquet(output / "candidate_oof_market_regimes.parquet")
    expected = candidates.loc[candidates["__ts__"].lt(pd.Timestamp("2025-02-15T00:00:00Z"))].reset_index(drop=True)
    posterior = [f"market_regime__state_p_{index}" for index in range(5)]
    phase = [f"market_regime__phase_p_{name}" for name in ("stable", "onset", "active", "settling")]
    assert np.allclose(timeline[posterior].sum(axis=1), 1.0, atol=1e-6)
    assert np.allclose(timeline[phase].sum(axis=1), 1.0, atol=1e-6)
    semantic = [f"regime_p_{name}" for name in PRIMARY_SEMANTIC_STATE_NAMES]
    assert np.allclose(timeline[semantic].sum(axis=1), 1.0, atol=1e-6)
    assert set(semantic).issubset(candidate)
    assert {"regime_entropy", "regime_top2_margin", "state_age_hours", "state_switch_probability", "market_direction_sign"}.issubset(timeline)
    # Every specialist exposes a padded, candidate-joinable soft simplex.  K
    # is fit label-free per OOF fold, so state_count records the live support;
    # zero padding must not change posterior mass.
    for system in ("trend_volatility", "breadth_dependence", "leverage_flow", "liquidity"):
        probabilities = [f"geometry_regime__{system}__state_p_{index}" for index in range(6)]
        assert set(probabilities).issubset(timeline)
        assert np.allclose(timeline[probabilities].sum(axis=1), 1.0, atol=1e-6)
        assert timeline[f"geometry_regime__{system}__state_count"].between(3, 6).all()
    relationship_break = "continuous_regime__relationship_break__trend_breadth__residual_signed_30d"
    assert relationship_break in timeline
    assert timeline[relationship_break].dtype == np.dtype("float32")
    assert len(candidate) == len(expected)
    assert candidate["candidate_id"].tolist() == expected["candidate_id"].tolist()
    assert (pd.to_datetime(candidate["regime_train_end_utc"], utc=True) < pd.to_datetime(candidate["__ts__"], utc=True)).all()
    assert (pd.to_datetime(candidate["transition_available_utc"], utc=True) <= pd.to_datetime(candidate["__ts__"], utc=True)).all()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["contract"]["primary"]["requested_state_count"] == 5
    assert manifest["contract"]["primary"]["postfit_low_support_merge"] is False
    assert relationship_break in manifest["contract"]["relationship_breaks"]["output_features"]
    assert manifest["coverage"]["candidate_rows"] == len(expected)


def test_primary_discovery_proxy_prefers_level_fields_over_accelerations(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "source_utc": pd.date_range("2025-01-01", periods=8, freq="h", tz="UTC"),
            "mv__trend_strength__level_6h": np.arange(8, dtype=np.float32),
            "mv__trend_strength__delta_6h": np.arange(8, dtype=np.float32),
            "mv__liquidity__spread_level_6h": np.arange(8, dtype=np.float32),
            "mv__liquidity__spread_delta_6h": np.arange(8, dtype=np.float32),
        }
    )
    path = tmp_path / "levels.parquet"
    frame.to_parquet(path, index=False)
    columns = compact_observable_columns(path, max_per_view=8)
    assert columns.index("mv__trend_strength__level_6h") < columns.index(
        "mv__trend_strength__delta_6h"
    )
    assert columns.index("mv__liquidity__spread_level_6h") < columns.index(
        "mv__liquidity__spread_delta_6h"
    )
