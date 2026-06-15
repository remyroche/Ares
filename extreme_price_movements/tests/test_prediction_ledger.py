import pandas as pd

from extreme_price_movements.inference.prediction_ledger import PredictionLedger


def test_prediction_ledger_appends_and_loads_unresolved(tmp_path):
    path = tmp_path / "prediction_ledger.parquet"
    ledger = PredictionLedger(path)
    ledger.append_rows(
        [
            {
                "timestamp": "2026-05-10T12:00:00Z",
                "symbol": "BTC/USDC",
                "strategy_id": "long_test",
                "normalized_rank_score": 0.95,
                "outcome_status": None,
            }
        ]
    )

    unresolved = ledger.load_unresolved(max_age_hours=10_000)
    assert len(unresolved) == 1
    assert unresolved.iloc[0]["symbol"] == "BTC/USDC"


def test_prediction_ledger_exposes_prefixed_lgbm_diagnostic_columns(tmp_path):
    path = tmp_path / "prediction_ledger.parquet"
    ledger = PredictionLedger(path)
    ledger.append_rows(
        [
            {
                "timestamp": "2026-05-10T12:00:00Z",
                "symbol": "BTC/USDC",
                "strategy_id": "short_test",
            }
        ]
    )

    df = pd.read_parquet(path)
    for prefix in ("base_lgbm", "meta_lgbm"):
        assert f"{prefix}_feature_drift_cov_shift" in df.columns
        assert f"{prefix}_regime_centroid_similarity_train" in df.columns
        assert f"{prefix}_prob_uncertainty" in df.columns


def test_prediction_ledger_marks_resolved(tmp_path):
    path = tmp_path / "prediction_ledger.parquet"
    ledger = PredictionLedger(path)
    ledger.append_rows(
        [
            {"timestamp": "2026-05-10T12:00:00Z", "symbol": "BTC/USDC"},
            {"timestamp": "2026-05-10T12:00:00Z", "symbol": "ETH/USDC"},
        ]
    )
    ledger.mark_resolved(
        pd.DataFrame(
            [
                {
                    "timestamp": "2026-05-10T12:00:00Z",
                    "symbol": "BTC/USDC",
                    "outcome_status": "resolved",
                    "tp_hit": True,
                }
            ]
        )
    )

    df = pd.read_parquet(path)
    btc = df.loc[df["symbol"] == "BTC/USDC"].iloc[0]
    eth = df.loc[df["symbol"] == "ETH/USDC"].iloc[0]
    assert btc["outcome_status"] == "resolved"
    assert bool(btc["tp_hit"]) is True
    assert pd.isna(eth.get("outcome_status"))
