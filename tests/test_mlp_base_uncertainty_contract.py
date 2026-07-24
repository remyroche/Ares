from __future__ import annotations

import pandas as pd

from scripts.run_meta_market_state_encoder_ablation import (
    _merge_base_predictive_uncertainty,
)


def test_base_predictive_uncertainty_join_is_keyed_and_audited(tmp_path) -> None:
    source = tmp_path / "base_uncertainty.parquet"
    rows = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(["2026-06-01T00:00:00Z"]),
            "__symbol__": ["BTC/USD:USD"],
            "side_name": ["long"],
            "base_lgbm_prob_std": [0.12],
            "base_lgbm_prob_uncertainty": [0.42],
            "realized_bad_mae": [1],
        }
    )
    rows.to_parquet(source, index=False)
    frame = rows.loc[:, ["__ts__", "__symbol__", "side_name"]].copy()

    out, audit = _merge_base_predictive_uncertainty(frame, source)

    assert out.loc[0, "base_lgbm_prob_std"] == 0.12
    assert out.loc[0, "base_lgbm_prob_uncertainty"] == 0.42
    assert "realized_bad_mae" not in out.columns
    assert audit["coverage"]["base_lgbm_prob_std"] == 1.0
