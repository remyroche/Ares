import sqlite3

import pandas as pd

from extreme_price_movements.inference.trade_logger import TradeLogger


def test_trade_logger_includes_required_parity_fields(tmp_path):
    path = tmp_path / "trades.csv"
    logger = TradeLogger(output_path=str(path), run_id="r1")

    decision = {
        "symbol": "BTC/USDT",
        "side": "long",
        "action": "enter",
        "strategy_id": "long_mr",
        "calibrated_score": 0.72,
        "rank_threshold": 0.60,
        "stop_price": 98.0,
        "stop_price_updated": 99.0,
        "orderbook_snapshot": "top5",
        "net_pnl": 0.01,
    }
    model_results = {"meta_pred": 0.8, "position_size": 0.5, "alpha_preds": {}}
    market_data = {"close": 100.0, "volume": 1234.0}
    cfg = {"run_id": "r1", "mode": "shadow"}

    logger.log_trade(decision, model_results, market_data, cfg)
    df = pd.read_csv(path)

    required = {
        "strategy_id",
        "calibrated_score",
        "rank_threshold",
        "stop_price",
        "stop_price_updated",
        "orderbook_snapshot",
        "net_pnl",
    }
    assert required.issubset(df.columns)
    assert df.loc[0, "strategy_id"] == "long_mr"

    db_path = path.with_suffix(".sqlite")
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT strategy_id, orderbook_snapshot, net_pnl FROM trades"
        ).fetchall()
    assert rows == [("long_mr", "top5", "0.01")]
