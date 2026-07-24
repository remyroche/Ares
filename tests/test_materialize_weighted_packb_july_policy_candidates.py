from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_weighted_packb_july_policy_candidates import materialize


def _reference_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    start = pd.Timestamp("2026-01-01", tz="UTC")
    for index in range(120):
        side = "long" if index % 2 == 0 else "short"
        score = index / 119.0
        rows.append(
            {
                "__ts__": start + pd.Timedelta(hours=index),
                "__symbol__": f"SYM{index % 7}",
                "side_name": side,
                "score_meta_base_soft_label": score,
                "ev_after_1pct": -0.01 + 0.04 * score + (0.003 if side == "long" else 0.0),
                "__first_touch_round_trip_cost__": 0.01,
                "archetype_label_family": "trend" if index % 3 else "reversal",
            }
        )
    return pd.DataFrame(rows)


def test_materialize_uses_only_prejuly_reference_and_preserves_execution_spread(
    tmp_path: Path,
) -> None:
    reference_path = tmp_path / "reference.parquet"
    prediction_path = tmp_path / "predictions.parquet"
    output_dir = tmp_path / "out"
    _reference_rows().to_parquet(reference_path, index=False)
    pd.DataFrame(
        [
            {
                "__ts__": "2026-07-01T00:00:00Z",
                "__symbol__": "BTC/USD:USD",
                "side_name": "long",
                "score_meta_base_soft_label": 0.99,
                "archetype_label_family": "trend",
                "policy_archetype": "must_not_win_fallback",
                "median_spread_bps": 14.0,
                "__barrier_pct__": 0.02,
            },
            {
                "__ts__": "2026-07-01T01:00:00Z",
                "__symbol__": "ETH/USD:USD",
                "side_name": "short",
                "score_meta_base_soft_label": 0.01,
                "policy_archetype": "fallback_policy_arch",
                "local_side_archetype": "short__must_not_win_local",
                "source_archetype": "must_not_win_source",
                "median_spread_bps": 22.0,
                "__barrier_pct__": 0.03,
            },
        ]
    ).to_parquet(prediction_path, index=False)

    manifest = materialize(
        predictions_path=prediction_path,
        calibration_reference_path=reference_path,
        output_dir=output_dir,
        min_side_rows=20,
        min_local_rows=15,
    )

    candidates = pd.read_parquet(output_dir / "weighted_packb_july_policy_candidates.parquet")
    high = candidates.loc[candidates["symbol"].eq("BTC/USD:USD")].iloc[0]
    low = candidates.loc[candidates["symbol"].eq("ETH/USD:USD")].iloc[0]
    assert high["base_archetype"] == "trend"
    assert low["base_archetype"] == "fallback_policy_arch"
    assert bool(high["policy_admitted_before_portfolio"])
    assert not bool(low["policy_admitted_before_portfolio"])
    assert float(high["policy_admission_rank"]) >= 0.90
    assert float(low["policy_admission_rank"]) == 0.0
    assert float(high["expected_spread_bps"]) == 14.0
    assert float(high["expected_half_spread_bps"]) == 7.0
    assert float(high["target_ev_embedded_round_trip_fee_bps"]) == 100.0
    assert not bool(high["target_ev_includes_spread"])
    assert manifest["reference_end"] < manifest["prediction_start"]
    assert manifest["cost_contract"]["fee_deducted_by_materializer"] is False
