import pandas as pd

from extreme_price_movements.inference.google_sheets_exporter import (
    CLOSED_TRADES_SHEET,
    GoogleSheetsTradeExporter,
    OPEN_TRADES_SHEET,
    STRATEGY_METRICS_SHEET,
    build_google_sheets_trade_tables,
)


def test_google_sheets_export_tables_split_open_closed_and_compute_unrealized():
    logs = pd.DataFrame(
        [
            {
                "timestamp": "2026-05-12T10:00:00Z",
                "position_id": "pos-open",
                "lifecycle_event": "entry_placed",
                "action": "enter",
                "status": "pending",
                "symbol": "BTC/USDC",
                "side": "long",
                "strategy_id": "s_long",
                "expected_entry_price": "9990",
                "realized_entry_price": "10000",
                "holding_time_hours": "1.25",
                "entry_notional_quote": "100",
                "requested_base_amount": "0.01",
                "effective_position_leverage": "2",
                "meta_pred": "0.91",
                "calibrated_score": "0.91",
                "rank_percentile": "0.97",
                "deployment_rank_threshold": "0.90",
            },
            {
                "timestamp": "2026-05-12T09:00:00Z",
                "position_id": "pos-closed",
                "lifecycle_event": "entry_placed",
                "action": "enter",
                "status": "pending",
                "symbol": "ETH/USDC",
                "side": "short",
                "strategy_id": "s_short",
                "realized_entry_price": "2000",
                "entry_notional_quote": "100",
            },
            {
                "timestamp": "2026-05-12T11:00:00Z",
                "position_id": "pos-closed",
                "lifecycle_event": "exit_filled",
                "action": "exit",
                "status": "closed",
                "symbol": "ETH/USDC",
                "side": "short",
                "strategy_id": "s_short",
                "entry_time": "2026-05-12T09:00:00Z",
                "exit_time": "2026-05-12T11:30:00Z",
                "exit_reason": "stop_loss_filled:original_stop_loss",
                "realized_entry_price": "2000",
                "realized_exit_price": "1990",
                "entry_notional_quote": "100",
                "net_pnl_amount": "0.4",
                "net_pnl_pct": "0.004",
                "net_pnl_pct_wallet": "0.008",
                "expected_hit_rate": "0.6",
                "mfe": "0.01",
                "mae": "-0.002",
            },
        ]
    )

    tables = build_google_sheets_trade_tables(
        logs,
        active_positions={
            "BTC/USDC": {
                "side": "long",
                "entry_price": 10000.0,
                "size": 0.01,
                "current_price": 10100.0,
                "mfe": 0.012,
                "mae": -0.001,
                "stop_price": 9900.0,
                "stop_reason": "original_stop_loss",
            }
        },
    )

    open_trades = tables[OPEN_TRADES_SHEET]
    closed_trades = tables[CLOSED_TRADES_SHEET]
    strategy_metrics = tables[STRATEGY_METRICS_SHEET]

    assert list(open_trades["position_id"]) == ["pos-open"]
    assert "holding_time_hours" in open_trades.columns
    assert float(open_trades.iloc[0]["holding_time_hours"]) == 1.25
    assert float(open_trades.iloc[0]["time_in_trade_hours"]) == 1.25
    assert float(open_trades.iloc[0]["current_unrealized_pnl"]) == 1.0
    assert float(open_trades.iloc[0]["current_unrealized_pnl_x_leverage"]) == 0.02

    assert list(closed_trades["position_id"]) == ["pos-closed"]
    assert closed_trades.iloc[0]["exit_reason"] == "stop_loss_filled:original_stop_loss"
    assert float(closed_trades.iloc[0]["holding_time_hours"]) == 2.5

    short_metrics = strategy_metrics[
        (strategy_metrics["strategy_id"] == "s_short")
        & (strategy_metrics["window_days"] == 3)
    ].iloc[0]
    assert int(short_metrics["closed_trades"]) == 1
    assert float(short_metrics["hit_rate"]) == 1.0
    assert float(short_metrics["surprise_hit_rate"]) == 0.4


def test_google_sheets_exporter_enabled_by_default_when_spreadsheet_id_configured(
    monkeypatch,
):
    monkeypatch.delenv("EPM_GOOGLE_SHEETS_ENABLED", raising=False)
    exporter = GoogleSheetsTradeExporter.from_config(
        {"google_spreadsheet_id": "spreadsheet-123"}
    )

    assert exporter is not None
    assert exporter.enabled is True

    disabled = GoogleSheetsTradeExporter.from_config(
        {
            "google_spreadsheet_id": "spreadsheet-123",
            "google_sheets_export_enabled": False,
        }
    )
    assert disabled is not None
    assert disabled.enabled is False


def test_export_task_to_sheet_missing_config(monkeypatch, capsys):
    from infrastructure.utils.google_sheets_exporter import export_task_to_sheet

    monkeypatch.delenv("SHEETS_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("SHEETS_WEBHOOK_SECRET", raising=False)

    assert export_task_to_sheet("sheet-123", "task", "ok", "job-1") is False
    assert "Missing env var: SHEETS_WEBHOOK_URL" in capsys.readouterr().out


def test_export_task_to_sheet_posts_payload_and_requires_ok(monkeypatch):
    from infrastructure.utils import google_sheets_exporter as exporter

    calls = []

    class FakeResponse:
        text = '{"ok": false}'

        def raise_for_status(self):
            return None

        def json(self):
            return {"ok": True}

    def fake_post(url, *, json, timeout):
        calls.append((url, json, timeout))
        return FakeResponse()

    monkeypatch.setenv("SHEETS_WEBHOOK_URL", "https://script.example/exec")
    monkeypatch.setenv("SHEETS_WEBHOOK_SECRET", "top-secret")
    monkeypatch.setattr(exporter.requests, "post", fake_post)

    assert exporter.export_task_to_sheet("sheet-123", "task", "ok", "job-1") is True
    assert calls == [
        (
            "https://script.example/exec",
            {
                "secret": "top-secret",
                "sheet_id": "sheet-123",
                "job_id": "job-1",
                "task": "task",
                "status": "ok",
            },
            10,
        )
    ]


def test_export_task_to_sheet_returns_false_when_webhook_rejects(monkeypatch, capsys):
    from infrastructure.utils import google_sheets_exporter as exporter

    class FakeResponse:
        text = '{"ok": false}'

        def raise_for_status(self):
            return None

        def json(self):
            return {"ok": False, "error": "unauthorized"}

    monkeypatch.setenv("SHEETS_WEBHOOK_URL", "https://script.example/exec")
    monkeypatch.setenv("SHEETS_WEBHOOK_SECRET", "top-secret")
    monkeypatch.setattr(
        exporter.requests, "post", lambda *args, **kwargs: FakeResponse()
    )

    assert exporter.export_task_to_sheet("sheet-123", "task", "ok", "job-1") is False
    output = capsys.readouterr().out
    assert "Apps Script rejected export" in output
    assert "top-secret" not in output
