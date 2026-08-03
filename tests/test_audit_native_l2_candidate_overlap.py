from __future__ import annotations

import pandas as pd

from scripts.audit_native_l2_candidate_overlap import _asof_join, _normalise_symbol


def test_native_and_candidate_product_identity_preserves_collateral() -> None:
    assert _normalise_symbol("AAVE_USD_USD") == "AAVE/USD:USD"
    assert _normalise_symbol("BTC_USD_BTC") == "BTC/USD:BTC"
    assert _normalise_symbol("AAVE_USD:USD") == "AAVE/USD:USD"
    assert _normalise_symbol("BTC/USD:USD") == "BTC/USD:USD"


def test_asof_join_never_uses_future_snapshot_and_keeps_unmatched_rows() -> None:
    panel = pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c"],
            "source_symbol": ["AAVE/USD:USD", "AAVE/USD:USD", "OTHER/USD:USD"],
            "candidate_ts": pd.to_datetime(
                ["2026-07-11T00:00:00Z", "2026-07-11T01:00:00Z", "2026-07-11T01:00:00Z"],
                utc=True,
            ),
        }
    )
    l2 = pd.DataFrame(
        {
            "symbol_norm": ["AAVE/USD:USD", "AAVE/USD:USD"],
            "snapshot_ts": pd.to_datetime(
                ["2026-07-11T00:30:00Z", "2026-07-11T00:45:00Z"], utc=True
            ),
            "feature_available_at": pd.to_datetime(
                ["2026-07-11T00:30:00Z", "2026-07-11T00:45:00Z"], utc=True
            ),
            "lag_features_ready": [False, True],
        }
    )
    joined = _asof_join(panel.sort_values(["candidate_ts", "source_symbol"]), l2)
    assert joined.loc[joined.candidate_id.eq("a"), "native_snapshot_match"].item() is False
    assert joined.loc[joined.candidate_id.eq("b"), "snapshot_ts"].item() == pd.Timestamp(
        "2026-07-11T00:45:00Z"
    )
    assert joined.loc[joined.candidate_id.eq("c"), "native_snapshot_match"].item() is False
    assert (joined.loc[joined.native_snapshot_match, "snapshot_ts"] <= joined.loc[joined.native_snapshot_match, "candidate_ts"]).all()
