import json
from pathlib import Path

import pandas as pd

from extreme_price_movements.inference.daily_reporter import (
    DailyDeploymentReporter,
    _confidence_calibration_recap,
)
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

        def fetch_balance(self, params=None):
            self.balance_params = params
            return {
                "total": {"USDC": 1100.0},
                "free": {"USDC": 1000.0},
                "used": {"USDC": 100.0},
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


def test_confidence_calibration_recap_empty_trades():
    assert (
        _confidence_calibration_recap(pd.DataFrame())
        == "insufficient closed trades for confidence/outcome recap"
    )


def test_confidence_calibration_recap_no_closed_trades():
    trades = pd.DataFrame(
        {
            "status": ["pending", "recorded"],
            "rank_percentile": [0.9, 0.7],
            "net_pnl_pct": [0.02, -0.01],
        }
    )

    assert (
        _confidence_calibration_recap(trades)
        == "insufficient closed trades for confidence/outcome recap"
    )


def test_confidence_calibration_recap_rank_percentile_and_net_pnl_pct():
    trades = pd.DataFrame(
        {
            "status": ["closed", "closed", "closed"],
            "rank_percentile": [0.40, 0.85, 0.96],
            "net_pnl_pct": [-0.01, 0.02, 0.04],
            "net_pnl_amount": [-1.0, 2.0, 4.0],
            "mfe": [0.01, 0.03, 0.05],
            "mae": [-0.02, -0.01, -0.005],
        }
    )

    recap = _confidence_calibration_recap(trades)

    assert "confidence_source=rank_percentile" in recap
    assert "closed_trades=3" in recap
    assert "hit_rate=66.6667%" in recap
    assert "avg_mfe=3.0000%" in recap
    assert "Monotonicity:" in recap
    assert "Probability calibration:" not in recap


def test_confidence_calibration_recap_bucket_outperformance_calculation():
    trades = pd.DataFrame(
        {
            "status": ["closed", "closed", "closed", "closed"],
            "rank_percentile": [0.45, 0.75, 0.90, 0.97],
            "net_pnl_pct": [-0.02, 0.00, 0.02, 0.10],
            "net_pnl_amount": [-2.0, 0.0, 2.0, 10.0],
        }
    )

    recap = _confidence_calibration_recap(trades)

    assert ">=0.95 | 1 | 97.0000% | 100.0000% | 10.0000% | 10.0000 | 7.5000%" in recap
    assert "Verdict: top-confidence trades outperformed by 7.5000% per trade" in recap


def test_confidence_calibration_recap_falls_back_to_calibrated_score():
    trades = pd.DataFrame(
        {
            "status": ["closed", "closed", "closed"],
            "rank_percentile": [None, None, None],
            "calibrated_score": [0.20, 0.80, 0.95],
            "net_pnl_pct": [-0.02, 0.03, 0.04],
            "net_pnl_amount": [-2.0, 3.0, 4.0],
        }
    )

    recap = _confidence_calibration_recap(trades)

    assert "confidence_source=calibrated_score" in recap
    assert "Probability calibration:" in recap
    assert "expected_hit_rate=0.650" in recap
    assert "realised_hit_rate=0.667" in recap
    assert "brier=" in recap


def test_confidence_calibration_recap_uses_net_amount_and_notional_fallback():
    trades = pd.DataFrame(
        {
            "status": ["closed", "closed"],
            "rank_percentile": [0.40, 0.97],
            "net_pnl_pct": [None, None],
            "net_pnl_amount": [-5.0, 20.0],
            "ridge_position_size": [100.0, 200.0],
        }
    )

    recap = _confidence_calibration_recap(trades)

    assert "closed_trades=2" in recap
    assert "avg_net_pnl_pct=2.5000%" in recap
    assert ">=0.95 | 1 | 97.0000% | 100.0000% | 10.0000% | 20.0000 | 7.5000%" in recap


def test_daily_report_body_includes_confidence_calibration_recap():
    reporter = DailyDeploymentReporter(state_path="unused")
    body = reporter._build_body(
        now=pd.Timestamp("2026-01-02T00:01:00Z"),
        total_balance=1000.0,
        previous_best_balance=950.0,
        amount_to_save=2.5,
        transfer_result={"success": True, "skipped": True},
        trades=pd.DataFrame(
            {
                "status": ["closed"],
                "rank_percentile": [0.96],
                "net_pnl_pct": [0.01],
                "net_pnl_amount": [1.0],
            }
        ),
    )

    assert "Confidence Calibration Recap" in body
    assert body.index("Model Drift And Execution") < body.index(
        "Confidence Calibration Recap"
    )
    assert body.index("Confidence Calibration Recap") < body.index("Net Strategy Recap")
