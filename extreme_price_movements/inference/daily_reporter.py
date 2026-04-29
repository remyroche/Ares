"""Daily deployment reporting and profit skim utilities."""

import json
import os
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from extreme_price_movements.inference.data_fetcher import classify_api_error
from extreme_price_movements.inference.trade_logger import TradeLogger
from extreme_price_movements.portfolio_manager import PortfolioManager
from extreme_price_movements.utils import tprint

DEFAULT_REPORT_TO = "cryptoalias.rp@proton.me"
DEFAULT_STATE_PATH = "extreme_price_movements/logs/daily_report_state.json"
REPORT_TRADE_COLUMNS = [
    "timestamp",
    "symbol",
    "side",
    "strategy_id",
    "meta_pred",
    "calibrated_score",
    "ridge_position_size",
    "entry_price",
    "actual_entry_price",
    "actual_exit_price",
    "net_pnl",
    "exit_reason",
    "status",
    "error",
]


def _load_dotenv_if_present(path: str = ".env") -> None:
    """Load missing environment variables from a simple .env file."""
    env_path = Path(path)
    if not env_path.exists():
        return
    try:
        for raw_line in env_path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except Exception as exc:
        tprint(f"[DailyReporter] Failed to load .env: {exc}")


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        tprint(f"[DailyReporter] Failed to read state {path}: {exc}")
        return {}


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)


def _coerce_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
        return out if np.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def _format_trade_report(trades: pd.DataFrame) -> str:
    if trades.empty:
        return "No trades logged since the previous daily message."
    cols = [col for col in REPORT_TRADE_COLUMNS if col in trades.columns]
    view = trades.loc[:, cols].tail(200).copy()
    for col in view.columns:
        view[col] = view[col].astype(str).str.slice(0, 120)
    return view.to_csv(index=False)


def _trades_since(logger: TradeLogger, since_ts: Optional[str]) -> pd.DataFrame:
    df = logger.read_logs()
    if df.empty or "timestamp" not in df.columns:
        return pd.DataFrame(columns=logger.columns)
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    valid = ts.notna()
    if since_ts:
        since = pd.Timestamp(since_ts)
        if since.tzinfo is None:
            since = since.tz_localize("UTC")
        else:
            since = since.tz_convert("UTC")
        valid &= ts > since
    return df.loc[valid].copy()


def transfer_profit_to_spot(
    exchange: Any,
    *,
    amount: float,
    asset: str = "USDT",
    transfer_type: str = "MARGIN_MAIN",
) -> Dict[str, Any]:
    """Transfer saved profit to spot through ccxt's Binance asset-transfer API."""
    amount_f = _coerce_float(amount, default=0.0)
    if amount_f <= 0.0:
        return {"success": True, "skipped": True, "reason": "zero_amount"}
    transfer = getattr(exchange, "sapiPostAssetTransfer", None)
    if not callable(transfer):
        transfer = getattr(exchange, "sapi_post_asset_transfer", None)
    if not callable(transfer):
        return {
            "success": False,
            "skipped": True,
            "error_category": "transfer_method_unavailable",
            "error": "exchange does not expose sapiPostAssetTransfer",
        }
    payload = {
        "type": str(transfer_type),
        "asset": str(asset),
        "amount": f"{amount_f:.8f}",
    }
    try:
        response = transfer(payload)
        return {
            "success": True,
            "skipped": False,
            "request": payload,
            "response": response,
        }
    except Exception as exc:
        return {
            "success": False,
            "skipped": False,
            "request": payload,
            "error_category": classify_api_error(exc),
            "error": str(exc),
        }


@dataclass
class DailyDeploymentReporter:
    """Run the daily balance checkpoint, profit skim, and Gmail report."""

    state_path: str = DEFAULT_STATE_PATH
    smtp_factory: Callable[..., Any] = smtplib.SMTP
    env_file: str = ".env"

    def _state(self) -> Dict[str, Any]:
        return _read_json(Path(self.state_path))

    def _save_state(self, state: Dict[str, Any]) -> None:
        _write_json_atomic(Path(self.state_path), state)

    def _send_email(
        self,
        *,
        subject: str,
        body: str,
        recipient: str,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        _load_dotenv_if_present(self.env_file)
        gmail_user = os.environ.get("GMAIL_USER", "").strip()
        gmail_password = os.environ.get("GMAIL_APP_PASSWORD", "").strip()
        smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com").strip()
        smtp_port = int(os.environ.get("SMTP_PORT", "587") or 587)
        if not gmail_user or not gmail_password:
            return {
                "success": False,
                "error_category": "missing_smtp_credentials",
                "error": "GMAIL_USER or GMAIL_APP_PASSWORD is missing",
            }

        message = EmailMessage()
        message["From"] = gmail_user
        message["To"] = recipient
        message["Subject"] = subject
        message.set_content(body)

        timeout = float(config.get("daily_report_smtp_timeout_seconds", 30.0))
        try:
            with self.smtp_factory(smtp_host, smtp_port, timeout=timeout) as smtp:
                smtp.starttls()
                smtp.login(gmail_user, gmail_password)
                smtp.send_message(message)
            return {"success": True, "recipient": recipient}
        except Exception as exc:
            return {
                "success": False,
                "error_category": classify_api_error(exc),
                "error": str(exc),
            }

    def _build_body(
        self,
        *,
        now: pd.Timestamp,
        total_balance: float,
        previous_best_balance: float,
        amount_to_save: float,
        transfer_result: Dict[str, Any],
        trades: pd.DataFrame,
    ) -> str:
        return "\n".join(
            [
                "Extreme price movement deployment daily report",
                "",
                f"datetime: {now.isoformat()}",
                f"total_balance_usdt: {total_balance:.8f}",
                f"previous_best_balance_usdt: {previous_best_balance:.8f}",
                f"amount_saved_to_spot_usdt: {amount_to_save:.8f}",
                f"transfer_result: {json.dumps(transfer_result, default=str, sort_keys=True)}",
                "",
                "Trades since previous message:",
                _format_trade_report(trades),
            ]
        )

    def maybe_run(
        self,
        *,
        exchange: Any,
        portfolio_mgr: PortfolioManager,
        trade_logger: TradeLogger,
        config: Optional[Dict[str, Any]] = None,
        now: Optional[pd.Timestamp] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Run the daily report if due; returns an execution summary."""
        cfg = dict(config or {})
        interval_hours = float(cfg.get("daily_report_interval_hours", 24.0))
        now_ts = pd.Timestamp(now or pd.Timestamp.now(tz="UTC"))
        if now_ts.tzinfo is None:
            now_ts = now_ts.tz_localize("UTC")
        else:
            now_ts = now_ts.tz_convert("UTC")

        state = self._state()
        last_report = state.get("last_report_ts")
        if last_report and not force:
            last_ts = pd.Timestamp(last_report)
            if last_ts.tzinfo is None:
                last_ts = last_ts.tz_localize("UTC")
            else:
                last_ts = last_ts.tz_convert("UTC")
            elapsed_hours = (now_ts - last_ts).total_seconds() / 3600.0
            if elapsed_hours < interval_hours:
                return {
                    "sent": False,
                    "reason": "not_due",
                    "elapsed_hours": elapsed_hours,
                }

        snapshot = portfolio_mgr.fetch_exchange_snapshot(exchange)
        total_balance = _coerce_float(snapshot.get("total_balance"))
        if not np.isfinite(total_balance):
            tprint("[DailyReporter] Skipping daily report: total balance unavailable")
            return {
                "sent": False,
                "reason": "balance_unavailable",
                "snapshot_errors": snapshot.get("errors", []),
            }

        previous_best = _coerce_float(
            state.get("previous_best_balance_usdt"), default=total_balance
        )
        amount_to_save = max(0.0, (total_balance - previous_best) / 20.0)
        transfer_enabled = bool(
            cfg.get("daily_report_transfer_enabled", cfg.get("mode") == "live")
        )
        transfer_result: Dict[str, Any]
        if transfer_enabled:
            transfer_result = transfer_profit_to_spot(
                exchange,
                amount=amount_to_save,
                transfer_type=str(cfg.get("daily_report_transfer_type", "MARGIN_MAIN")),
            )
        else:
            transfer_result = {
                "success": True,
                "skipped": True,
                "reason": "transfer_disabled",
            }

        if (
            transfer_enabled
            and amount_to_save > 0.0
            and bool(transfer_result.get("success"))
        ):
            # Persist the new high-water mark immediately after a successful
            # transfer so an SMTP failure cannot repeat the same transfer.
            state = dict(state)
            state["previous_best_balance_usdt"] = max(previous_best, total_balance)
            state["last_total_balance_usdt"] = total_balance
            state["last_amount_saved_to_spot_usdt"] = amount_to_save
            state["last_transfer_result"] = transfer_result
            state["last_transfer_ts"] = now_ts.isoformat()
            self._save_state(state)

        trades = _trades_since(trade_logger, state.get("last_trade_report_ts"))
        recipient = str(
            cfg.get("daily_report_email_to")
            or os.environ.get("EPM_REPORT_EMAIL_TO")
            or DEFAULT_REPORT_TO
        )
        subject = cfg.get("daily_report_subject", "EPM daily deployment report")
        body = self._build_body(
            now=now_ts,
            total_balance=total_balance,
            previous_best_balance=previous_best,
            amount_to_save=amount_to_save,
            transfer_result=transfer_result,
            trades=trades,
        )
        email_result = self._send_email(
            subject=str(subject), body=body, recipient=recipient, config=cfg
        )
        if not email_result.get("success"):
            tprint(
                "[DailyReporter] Daily report email failed: "
                f"{email_result.get('error_category')}: {email_result.get('error')}"
            )
            return {
                "sent": False,
                "reason": "email_failed",
                "email_result": email_result,
                "transfer_result": transfer_result,
                "amount_to_save": amount_to_save,
            }

        new_state = dict(state)
        new_state["previous_best_balance_usdt"] = max(previous_best, total_balance)
        new_state["last_report_ts"] = now_ts.isoformat()
        new_state["last_trade_report_ts"] = now_ts.isoformat()
        new_state["last_total_balance_usdt"] = total_balance
        new_state["last_amount_saved_to_spot_usdt"] = amount_to_save
        new_state["last_transfer_result"] = transfer_result
        self._save_state(new_state)
        tprint(
            "[DailyReporter] Daily report sent: "
            f"total={total_balance:.2f} previous_best={previous_best:.2f} "
            f"saved={amount_to_save:.2f} trades={len(trades)}"
        )
        return {
            "sent": True,
            "email_result": email_result,
            "transfer_result": transfer_result,
            "amount_to_save": amount_to_save,
            "trade_count": int(len(trades)),
            "total_balance": total_balance,
            "previous_best_balance": previous_best,
        }
