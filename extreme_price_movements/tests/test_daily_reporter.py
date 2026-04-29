import json
from pathlib import Path

import pandas as pd

from extreme_price_movements.inference.daily_reporter import DailyDeploymentReporter
from extreme_price_movements.inference.trade_logger import TradeLogger
from extreme_price_movements.portfolio_manager import PortfolioManager


class _FakeSMTP:
    sent_messages = []
    logins = []

    def __init__(self, host, port, timeout=None):
        self.host = host
        self.port = port
        self.timeout = timeout

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def starttls(self):
        return None

    def login(self, user, password):
        self.logins.append((user, password))

    def send_message(self, message):
        self.sent_messages.append(message)


def test_daily_reporter_sends_balance_report_and_transfers_profit(
    tmp_path, monkeypatch
):
    _FakeSMTP.sent_messages = []
    _FakeSMTP.logins = []
    monkeypatch.setenv("GMAIL_USER", "sender@example.com")
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "app-password")
    monkeypatch.setenv("SMTP_HOST", "smtp.test")
    monkeypatch.setenv("SMTP_PORT", "2525")

    class _Exchange:
        def __init__(self):
            self.transfers = []

        def fetch_balance(self):
            return {
                "total": {"USDT": 1100.0},
                "free": {"USDT": 1000.0},
                "used": {"USDT": 100.0},
            }

        def sapiPostAssetTransfer(self, payload):
            self.transfers.append(payload)
            return {"tranId": "abc"}

    logger = TradeLogger(output_path=str(tmp_path / "trades.csv"), run_id="r1")
    logger.log_entry(
        symbol="BTC/USDT",
        side="long",
        size=100.0,
        price=50000.0,
        predictions={"meta_pred": 0.8},
        features={"strategy_id": "long_mr", "net_pnl": 12.5},
        mode="live",
    )

    state_path = tmp_path / "daily_state.json"
    state_path.write_text(
        json.dumps(
            {
                "previous_best_balance_usdt": 1000.0,
                "last_report_ts": "2026-01-01T00:00:00+00:00",
                "last_trade_report_ts": "2026-01-01T00:00:00+00:00",
            }
        )
    )
    exchange = _Exchange()
    reporter = DailyDeploymentReporter(
        state_path=str(state_path),
        smtp_factory=_FakeSMTP,
        env_file=str(tmp_path / ".env"),
    )

    result = reporter.maybe_run(
        exchange=exchange,
        portfolio_mgr=PortfolioManager(portfolio_value=1000.0),
        trade_logger=logger,
        config={
            "mode": "live",
            "daily_report_email_to": "ops@example.com",
            "daily_report_transfer_enabled": True,
        },
        now=pd.Timestamp("2026-01-02T00:01:00Z"),
    )

    assert result["sent"] is True
    assert result["amount_to_save"] == 5.0
    assert exchange.transfers == [
        {"type": "MARGIN_MAIN", "asset": "USDT", "amount": "5.00000000"}
    ]
    assert _FakeSMTP.logins == [("sender@example.com", "app-password")]
    assert len(_FakeSMTP.sent_messages) == 1
    body = _FakeSMTP.sent_messages[0].get_content()
    assert "total_balance_usdt: 1100.00000000" in body
    assert "previous_best_balance_usdt: 1000.00000000" in body
    assert "amount_saved_to_spot_usdt: 5.00000000" in body
    assert "BTC/USDT" in body

    state = json.loads(Path(state_path).read_text())
    assert state["previous_best_balance_usdt"] == 1100.0
    assert state["last_amount_saved_to_spot_usdt"] == 5.0


def test_daily_reporter_respects_interval(tmp_path, monkeypatch):
    monkeypatch.setenv("GMAIL_USER", "sender@example.com")
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "app-password")
    state_path = tmp_path / "daily_state.json"
    state_path.write_text(
        json.dumps(
            {
                "previous_best_balance_usdt": 1000.0,
                "last_report_ts": "2026-01-02T00:00:00+00:00",
            }
        )
    )

    reporter = DailyDeploymentReporter(
        state_path=str(state_path),
        smtp_factory=_FakeSMTP,
        env_file=str(tmp_path / ".env"),
    )
    result = reporter.maybe_run(
        exchange=object(),
        portfolio_mgr=PortfolioManager(),
        trade_logger=TradeLogger(output_path=str(tmp_path / "trades.csv")),
        now=pd.Timestamp("2026-01-02T01:00:00Z"),
    )

    assert result["sent"] is False
    assert result["reason"] == "not_due"
