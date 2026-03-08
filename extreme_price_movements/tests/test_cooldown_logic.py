import pandas as pd
import importlib.util
from pathlib import Path

from extreme_price_movements.ridge_position_sizer import _asset_overlap_keep_mask


_TRADE_LOGGER_PATH = (
    Path(__file__).resolve().parents[1] / "inference" / "trade_logger.py"
)
_TRADE_LOGGER_SPEC = importlib.util.spec_from_file_location(
    "trade_logger_local",
    _TRADE_LOGGER_PATH,
)
_TRADE_LOGGER_MODULE = importlib.util.module_from_spec(_TRADE_LOGGER_SPEC)
assert _TRADE_LOGGER_SPEC is not None and _TRADE_LOGGER_SPEC.loader is not None
_TRADE_LOGGER_SPEC.loader.exec_module(_TRADE_LOGGER_MODULE)
TradeLogger = _TRADE_LOGGER_MODULE.TradeLogger


def test_asset_overlap_keep_mask_enforces_cooldown_hours():
    ts = pd.to_datetime(
        [
            "2026-03-07T00:00:00Z",
            "2026-03-07T00:30:00Z",
            "2026-03-07T02:30:00Z",
            "2026-03-07T00:15:00Z",
        ],
        utc=True,
    )
    assets = ["BTC", "BTC", "BTC", "ETH"]
    keep = _asset_overlap_keep_mask(
        timestamps=ts.to_numpy(),
        assets=assets,
        exit_bars=None,
        priority=[1.0, 0.9, 0.8, 0.7],
        cooldown_hours=2.0,
    )
    assert keep.tolist() == [True, False, True, True]

def test_trade_logger_returns_last_trade_timestamp(tmp_path):
    log_path = tmp_path / "trades.csv"
    logger = TradeLogger(output_path=str(log_path), run_id="test")
    logger.log_trade(
        decision={"symbol": "BTC/USDT", "side": "long", "action": "enter", "status": "completed"},
        model_results={},
        market_data={"close": 100.0},
        config={"mode": "shadow", "run_id": "test"},
    )
    last_ts = logger.get_last_trade_timestamp("BTC/USDT")
    assert last_ts is not None
