"""Optional Google Sheets export for live inference trade reporting."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd

from extreme_price_movements.inference.trade_logger import TradeLogger
from extreme_price_movements.utils import tprint


OPEN_TRADES_SHEET = "Open Trades"
CLOSED_TRADES_SHEET = "Closed Trades"
STRATEGY_METRICS_SHEET = "Strategy Metrics"

OPEN_TRADE_COLUMNS = [
    "symbol",
    "side",
    "strategy_id",
    "entry_time",
    "time_in_trade_hours",
    "leverage",
    "entry_notional_quote",
    "size",
    "expected_entry_price",
    "realized_entry_price",
    "current_price",
    "current_unrealized_pnl",
    "current_unrealized_pnl_pct",
    "current_unrealized_pnl_x_leverage",
    "mfe",
    "mae",
    "meta_pred",
    "calibrated_score",
    "rank_percentile",
    "deployment_rank_threshold",
    "stop_price",
    "stop_reason",
    "last_update",
    "position_id",
]

CLOSED_TRADE_COLUMNS = [
    "exit_time",
    "entry_time",
    "symbol",
    "side",
    "strategy_id",
    "exit_reason",
    "exit_reason_detail",
    "entry_notional_quote",
    "exit_notional_quote",
    "expected_entry_price",
    "realized_entry_price",
    "realized_exit_price",
    "gross_pnl_amount",
    "net_pnl_amount",
    "gross_pnl_pct",
    "net_pnl_pct",
    "leverage_adjusted_net_pnl_pct",
    "net_pnl_pct_wallet",
    "mfe",
    "mae",
    "meta_pred",
    "calibrated_score",
    "rank_percentile",
    "deployment_rank_threshold",
    "expected_hit_rate",
    "realized_hit_rate",
    "calibration_error",
    "fees_amount",
    "realized_fee_bps",
    "realized_funding_bps",
    "realized_borrow_bps",
    "policy_parity_ok",
    "exit_vs_policy_stop_bps",
    "position_id",
]

STRATEGY_METRIC_COLUMNS = [
    "strategy_id",
    "window_days",
    "closed_trades",
    "notional_net_pnl",
    "notional_net_pnl_pct_sum",
    "notional_net_pnl_pct_x_leverage_sum",
    "bankroll_pnl_pct_sum",
    "hit_rate",
    "expected_hit_rate",
    "surprise_hit_rate",
    "sortino",
    "avg_net_pnl_pct",
    "avg_net_pnl_amount",
    "max_drawdown_amount",
    "last_exit_time",
]


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def _first_present(row: pd.Series, names: Iterable[str], default: Any = "") -> Any:
    for name in names:
        if name in row and pd.notna(row[name]) and str(row[name]) != "":
            return row[name]
    return default


def _parse_time(value: Any) -> pd.Timestamp:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    return pd.Timestamp(ts) if pd.notna(ts) else pd.NaT


def _stringify_frame(df: pd.DataFrame) -> list[list[Any]]:
    if df is None:
        return [[]]
    clean = df.copy()
    if clean.empty:
        return [list(clean.columns)] if len(clean.columns) else [[]]
    for col in clean.columns:
        if pd.api.types.is_datetime64_any_dtype(clean[col]):
            clean[col] = clean[col].dt.strftime("%Y-%m-%d %H:%M:%S%z")
    clean = clean.replace([np.inf, -np.inf], np.nan)
    clean = clean.where(pd.notna(clean), "")
    return [list(clean.columns)] + clean.astype(object).values.tolist()


def _is_entry_row(df: pd.DataFrame) -> pd.Series:
    action = df.get("action", pd.Series("", index=df.index)).astype(str).str.lower()
    event = (
        df.get("lifecycle_event", pd.Series("", index=df.index))
        .astype(str)
        .str.lower()
    )
    status = df.get("status", pd.Series("", index=df.index)).astype(str).str.lower()
    return (
        action.eq("enter")
        & ~event.str.contains("failed|rejected|absent", regex=True, na=False)
        & ~status.isin({"failed", "rejected", "error", "reconciled_absent"})
    )


def _is_exit_row(df: pd.DataFrame) -> pd.Series:
    action = df.get("action", pd.Series("", index=df.index)).astype(str).str.lower()
    event = (
        df.get("lifecycle_event", pd.Series("", index=df.index))
        .astype(str)
        .str.lower()
    )
    status = df.get("status", pd.Series("", index=df.index)).astype(str).str.lower()
    has_realized_exit = pd.to_numeric(
        df.get("realized_exit_price", pd.Series(np.nan, index=df.index)),
        errors="coerce",
    ).notna()
    return (
        action.eq("exit")
        | event.str.contains("exit|closed|close", regex=True, na=False)
        | status.isin({"closed", "completed"})
        | has_realized_exit
    )


def _latest_by_position(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    work = df.copy()
    work["_ts"] = pd.to_datetime(work.get("timestamp"), utc=True, errors="coerce")
    work["_pos"] = work.get("position_id", "").astype(str)
    work = work.sort_values("_ts")
    return work.drop_duplicates("_pos", keep="last").drop(columns=["_ts", "_pos"])


def _closed_position_ids(df: pd.DataFrame) -> set[str]:
    if df.empty or "position_id" not in df.columns:
        return set()
    exits = df[_is_exit_row(df)]
    return {str(v) for v in exits["position_id"].dropna() if str(v)}


def _position_state_by_id(active_positions: Optional[Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for symbol, state in (active_positions or {}).items():
        if not isinstance(state, dict):
            continue
        position_id = str(state.get("position_id") or "")
        if position_id:
            out[position_id] = state
        out.setdefault(str(symbol), state)
    return out


def build_open_trades_table(
    trade_logs: pd.DataFrame,
    *,
    active_positions: Optional[Dict[str, Dict[str, Any]]] = None,
) -> pd.DataFrame:
    """Build the open-trade sheet from trade logs plus live active position state."""
    if trade_logs is None or trade_logs.empty:
        return pd.DataFrame(columns=OPEN_TRADE_COLUMNS)
    entries = _latest_by_position(trade_logs[_is_entry_row(trade_logs)].copy())
    if entries.empty:
        return pd.DataFrame(columns=OPEN_TRADE_COLUMNS)
    closed_ids = _closed_position_ids(trade_logs)
    if "position_id" in entries.columns:
        entries = entries[~entries["position_id"].astype(str).isin(closed_ids)]
    state_by_id = _position_state_by_id(active_positions)
    rows = []
    now = pd.Timestamp.now(tz="UTC")
    for _, row in entries.iterrows():
        symbol = str(row.get("symbol") or "")
        position_id = str(row.get("position_id") or "")
        state = state_by_id.get(position_id) or state_by_id.get(symbol) or {}
        entry_price = _safe_float(
            state.get("entry_price"),
            _safe_float(_first_present(row, ["realized_entry_price", "entry_price"])),
        )
        current_price = _safe_float(
            state.get("current_price", state.get("last_price", state.get("mark_price"))),
            _safe_float(row.get("ticker_mid")),
        )
        side = str(state.get("side") or row.get("side") or "").lower()
        direction = 1.0 if side == "long" else -1.0
        size = _safe_float(state.get("size"), _safe_float(row.get("requested_base_amount")))
        if not np.isfinite(size) or size == 0.0:
            notional = _safe_float(row.get("entry_notional_quote"))
            size = notional / max(entry_price, 1e-12) if np.isfinite(notional) else np.nan
        unrealized = (
            direction * (current_price - entry_price) * size
            if np.isfinite(current_price) and np.isfinite(entry_price) and np.isfinite(size)
            else np.nan
        )
        entry_notional = _safe_float(
            row.get("entry_notional_quote"),
            entry_price * size if np.isfinite(entry_price) and np.isfinite(size) else np.nan,
        )
        unrealized_pct = (
            unrealized / max(abs(entry_notional), 1e-12)
            if np.isfinite(unrealized) and np.isfinite(entry_notional)
            else np.nan
        )
        leverage = _safe_float(
            _first_present(
                row,
                ["effective_position_leverage", "leverage_wallet_multiplier"],
                state.get("effective_position_leverage", state.get("leverage_wallet_multiplier", 1.0)),
            ),
            1.0,
        )
        entry_time = _parse_time(state.get("entry_time") or row.get("timestamp"))
        rows.append(
            {
                "symbol": symbol,
                "side": side or row.get("side"),
                "strategy_id": state.get("strategy_id") or state.get("bucket_key") or row.get("strategy_id"),
                "entry_time": entry_time,
                "time_in_trade_hours": (
                    float((now - entry_time).total_seconds() / 3600.0)
                    if pd.notna(entry_time)
                    else np.nan
                ),
                "leverage": leverage,
                "entry_notional_quote": entry_notional,
                "size": size,
                "expected_entry_price": row.get("expected_entry_price"),
                "realized_entry_price": row.get("realized_entry_price") or entry_price,
                "current_price": current_price,
                "current_unrealized_pnl": unrealized,
                "current_unrealized_pnl_pct": unrealized_pct,
                "current_unrealized_pnl_x_leverage": (
                    unrealized_pct * leverage
                    if np.isfinite(unrealized_pct) and np.isfinite(leverage)
                    else np.nan
                ),
                "mfe": _safe_float(state.get("mfe"), _safe_float(row.get("mfe"))),
                "mae": _safe_float(state.get("mae"), _safe_float(row.get("mae"))),
                "meta_pred": row.get("meta_pred"),
                "calibrated_score": row.get("calibrated_score"),
                "rank_percentile": row.get("rank_percentile"),
                "deployment_rank_threshold": row.get("deployment_rank_threshold"),
                "stop_price": state.get("stop_price") or row.get("stop_price"),
                "stop_reason": state.get("stop_reason") or row.get("exit_reason_detail"),
                "last_update": state.get("last_update") or row.get("timestamp"),
                "position_id": position_id,
            }
        )
    return pd.DataFrame(rows).reindex(columns=OPEN_TRADE_COLUMNS)


def build_closed_trades_table(trade_logs: pd.DataFrame) -> pd.DataFrame:
    if trade_logs is None or trade_logs.empty:
        return pd.DataFrame(columns=CLOSED_TRADE_COLUMNS)
    exits = trade_logs[_is_exit_row(trade_logs)].copy()
    if exits.empty:
        return pd.DataFrame(columns=CLOSED_TRADE_COLUMNS)
    exits["_exit_ts"] = pd.to_datetime(exits.get("timestamp"), utc=True, errors="coerce")
    exits = exits.sort_values("_exit_ts", ascending=False)
    rows = []
    for _, row in exits.iterrows():
        rows.append(
            {
                "exit_time": row.get("timestamp"),
                "entry_time": row.get("entry_time"),
                "symbol": row.get("symbol"),
                "side": row.get("side"),
                "strategy_id": row.get("strategy_id"),
                "exit_reason": row.get("exit_reason"),
                "exit_reason_detail": row.get("exit_reason_detail"),
                "entry_notional_quote": row.get("entry_notional_quote"),
                "exit_notional_quote": row.get("exit_notional_quote"),
                "expected_entry_price": row.get("expected_entry_price"),
                "realized_entry_price": row.get("realized_entry_price") or row.get("actual_entry_price"),
                "realized_exit_price": row.get("realized_exit_price") or row.get("actual_exit_price"),
                "gross_pnl_amount": row.get("gross_pnl_amount"),
                "net_pnl_amount": row.get("net_pnl_amount") or row.get("net_pnl"),
                "gross_pnl_pct": row.get("gross_pnl_pct"),
                "net_pnl_pct": row.get("net_pnl_pct"),
                "leverage_adjusted_net_pnl_pct": row.get("leverage_adjusted_net_pnl_pct"),
                "net_pnl_pct_wallet": row.get("net_pnl_pct_wallet"),
                "mfe": row.get("mfe"),
                "mae": row.get("mae"),
                "meta_pred": row.get("meta_pred"),
                "calibrated_score": row.get("calibrated_score"),
                "rank_percentile": row.get("rank_percentile"),
                "deployment_rank_threshold": row.get("deployment_rank_threshold"),
                "expected_hit_rate": row.get("expected_hit_rate"),
                "realized_hit_rate": row.get("realized_hit_rate"),
                "calibration_error": row.get("calibration_error"),
                "fees_amount": row.get("fees_amount"),
                "realized_fee_bps": row.get("realized_fee_bps"),
                "realized_funding_bps": row.get("realized_funding_bps"),
                "realized_borrow_bps": row.get("realized_borrow_bps"),
                "policy_parity_ok": row.get("policy_parity_ok"),
                "exit_vs_policy_stop_bps": row.get("exit_vs_policy_stop_bps"),
                "position_id": row.get("position_id"),
            }
        )
    return pd.DataFrame(rows).reindex(columns=CLOSED_TRADE_COLUMNS)


def _sortino(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return np.nan
    downside = vals[vals < 0.0]
    if downside.empty or float(np.sqrt(np.mean(np.square(downside)))) == 0.0:
        return 100.0 if float(vals.mean()) > 0.0 else 0.0
    return float(vals.mean() / np.sqrt(np.mean(np.square(downside))))


def _max_drawdown_amount(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").fillna(0.0)
    if vals.empty:
        return np.nan
    cum = vals.cumsum()
    dd = cum - cum.cummax()
    return float(dd.min()) if len(dd) else np.nan


def build_strategy_metrics_table(
    closed_trades: pd.DataFrame,
    *,
    windows: Iterable[int] = (3, 15, 30),
) -> pd.DataFrame:
    if closed_trades is None or closed_trades.empty:
        return pd.DataFrame(columns=STRATEGY_METRIC_COLUMNS)
    work = closed_trades.copy()
    work["_exit_ts"] = pd.to_datetime(work["exit_time"], utc=True, errors="coerce")
    work = work.dropna(subset=["_exit_ts"])
    if work.empty:
        return pd.DataFrame(columns=STRATEGY_METRIC_COLUMNS)
    now = pd.Timestamp.now(tz="UTC")
    work["net_pnl_amount_num"] = pd.to_numeric(work["net_pnl_amount"], errors="coerce")
    work["net_pnl_pct_num"] = pd.to_numeric(work["net_pnl_pct"], errors="coerce")
    lev_pct = pd.to_numeric(work["leverage_adjusted_net_pnl_pct"], errors="coerce")
    leverage = pd.to_numeric(work.get("leverage", pd.Series(np.nan, index=work.index)), errors="coerce")
    work["net_pnl_pct_x_leverage_num"] = lev_pct.where(
        lev_pct.notna(),
        work["net_pnl_pct_num"] * leverage.fillna(1.0),
    )
    bankroll = pd.to_numeric(work["net_pnl_pct_wallet"], errors="coerce")
    work["bankroll_pnl_pct_num"] = bankroll.where(bankroll.notna(), work["net_pnl_pct_x_leverage_num"])
    expected_hit = pd.to_numeric(work.get("expected_hit_rate", pd.Series(np.nan, index=work.index)), errors="coerce")
    rows = []
    for strategy_id, grp_all in work.groupby("strategy_id", dropna=False, sort=False):
        for days in windows:
            start = now - pd.Timedelta(days=int(days))
            grp = grp_all[grp_all["_exit_ts"] >= start].copy()
            if grp.empty:
                rows.append(
                    {
                        "strategy_id": strategy_id,
                        "window_days": int(days),
                        "closed_trades": 0,
                    }
                )
                continue
            wins = grp["net_pnl_amount_num"] > 0.0
            exp = expected_hit.loc[grp.index].dropna()
            hit_rate = float(wins.mean()) if len(grp) else np.nan
            expected_rate = float(exp.mean()) if not exp.empty else np.nan
            rows.append(
                {
                    "strategy_id": strategy_id,
                    "window_days": int(days),
                    "closed_trades": int(len(grp)),
                    "notional_net_pnl": float(grp["net_pnl_amount_num"].sum(skipna=True)),
                    "notional_net_pnl_pct_sum": float(grp["net_pnl_pct_num"].sum(skipna=True)),
                    "notional_net_pnl_pct_x_leverage_sum": float(
                        grp["net_pnl_pct_x_leverage_num"].sum(skipna=True)
                    ),
                    "bankroll_pnl_pct_sum": float(grp["bankroll_pnl_pct_num"].sum(skipna=True)),
                    "hit_rate": hit_rate,
                    "expected_hit_rate": expected_rate,
                    "surprise_hit_rate": (
                        hit_rate - expected_rate
                        if np.isfinite(hit_rate) and np.isfinite(expected_rate)
                        else np.nan
                    ),
                    "sortino": _sortino(grp["net_pnl_pct_num"]),
                    "avg_net_pnl_pct": float(grp["net_pnl_pct_num"].mean(skipna=True)),
                    "avg_net_pnl_amount": float(grp["net_pnl_amount_num"].mean(skipna=True)),
                    "max_drawdown_amount": _max_drawdown_amount(grp["net_pnl_amount_num"]),
                    "last_exit_time": grp["_exit_ts"].max(),
                }
            )
    return pd.DataFrame(rows).reindex(columns=STRATEGY_METRIC_COLUMNS)


def build_google_sheets_trade_tables(
    trade_logs: pd.DataFrame,
    *,
    active_positions: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, pd.DataFrame]:
    closed = build_closed_trades_table(trade_logs)
    return {
        OPEN_TRADES_SHEET: build_open_trades_table(
            trade_logs,
            active_positions=active_positions,
        ),
        CLOSED_TRADES_SHEET: closed,
        STRATEGY_METRICS_SHEET: build_strategy_metrics_table(closed),
    }


@dataclass
class GoogleSheetsTradeExporter:
    """Push trade reporting tables to a Google spreadsheet when configured."""

    spreadsheet_id: str
    credentials_path: Optional[str] = None
    service_account_json: Optional[str] = None
    enabled: bool = True
    min_interval_seconds: float = 300.0
    timeout_seconds: float = 30.0
    _last_export_monotonic: float = field(default=0.0, init=False)
    _warned_disabled: bool = field(default=False, init=False)
    _session: Any = field(default=None, init=False, repr=False)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GoogleSheetsTradeExporter | None":
        spreadsheet_id = str(
            config.get("google_spreadsheet_id")
            or os.environ.get("EPM_GOOGLE_SPREADSHEET_ID", "")
        ).strip()
        enabled_raw = (
            config.get("google_sheets_export_enabled")
            if "google_sheets_export_enabled" in config
            else os.environ.get("EPM_GOOGLE_SHEETS_ENABLED")
        )
        if enabled_raw is None or str(enabled_raw).strip() == "":
            enabled = bool(spreadsheet_id)
        else:
            enabled = str(enabled_raw).strip().lower() not in {
                "0",
                "false",
                "no",
                "off",
                "disabled",
            }
        if not enabled and not spreadsheet_id:
            return None
        if not spreadsheet_id:
            tprint("Google Sheets export enabled but EPM_GOOGLE_SPREADSHEET_ID is missing.")
            return None
        return cls(
            spreadsheet_id=spreadsheet_id,
            credentials_path=str(
                config.get("google_application_credentials")
                or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
            )
            or None,
            service_account_json=str(
                config.get("google_service_account_json")
                or os.environ.get("EPM_GOOGLE_SERVICE_ACCOUNT_JSON", "")
            )
            or None,
            enabled=enabled,
            min_interval_seconds=float(
                config.get(
                    "google_sheets_export_min_interval_seconds",
                    os.environ.get("EPM_GOOGLE_SHEETS_MIN_INTERVAL_SECONDS", 300.0),
                )
            ),
            timeout_seconds=float(
                config.get(
                    "google_sheets_export_timeout_seconds",
                    os.environ.get("EPM_GOOGLE_SHEETS_TIMEOUT_SECONDS", 30.0),
                )
            ),
        )

    def should_export(self, *, force: bool = False) -> bool:
        if not self.enabled:
            if not self._warned_disabled:
                tprint("Google Sheets exporter configured but disabled.")
                self._warned_disabled = True
            return False
        if force:
            return True
        now = time.monotonic()
        return (now - self._last_export_monotonic) >= float(self.min_interval_seconds)

    def _credentials(self):
        from google.oauth2 import service_account

        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        if self.service_account_json:
            info_text = self.service_account_json
            if info_text.lstrip().startswith("{"):
                info = json.loads(info_text)
            elif Path(info_text).exists():
                info = json.loads(Path(info_text).read_text())
            else:
                info = json.loads(info_text)
            return service_account.Credentials.from_service_account_info(
                info,
                scopes=scopes,
            )
        if self.credentials_path:
            return service_account.Credentials.from_service_account_file(
                self.credentials_path,
                scopes=scopes,
            )
        raise RuntimeError(
            "Google Sheets export requires GOOGLE_APPLICATION_CREDENTIALS or "
            "EPM_GOOGLE_SERVICE_ACCOUNT_JSON."
        )

    def _authorized_session(self):
        if self._session is not None:
            return self._session
        from google.auth.transport.requests import AuthorizedSession

        self._session = AuthorizedSession(self._credentials())
        return self._session

    def _api_base(self) -> str:
        return f"https://sheets.googleapis.com/v4/spreadsheets/{self.spreadsheet_id}"

    def _existing_sheets(self) -> set[str]:
        session = self._authorized_session()
        response = session.get(
            self._api_base(),
            params={"fields": "sheets.properties.title"},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        return {
            str(sheet.get("properties", {}).get("title", ""))
            for sheet in payload.get("sheets", [])
        }

    def _ensure_sheets(self, sheet_names: Iterable[str]) -> None:
        existing = self._existing_sheets()
        missing = [name for name in sheet_names if name not in existing]
        if not missing:
            return
        requests = [{"addSheet": {"properties": {"title": name}}} for name in missing]
        session = self._authorized_session()
        response = session.post(
            self._api_base() + ":batchUpdate",
            json={"requests": requests},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()

    def _update_sheet(self, sheet_name: str, df: pd.DataFrame) -> None:
        session = self._authorized_session()
        escaped = sheet_name.replace("'", "''")
        range_name = f"'{escaped}'!A1"
        clear = session.post(
            self._api_base() + f"/values/{range_name}:clear",
            timeout=self.timeout_seconds,
        )
        clear.raise_for_status()
        values = _stringify_frame(df)
        update = session.put(
            self._api_base() + f"/values/{range_name}",
            params={"valueInputOption": "RAW"},
            json={"range": range_name, "majorDimension": "ROWS", "values": values},
            timeout=self.timeout_seconds,
        )
        update.raise_for_status()

    def export_tables(self, tables: Dict[str, pd.DataFrame], *, force: bool = False) -> bool:
        if not self.should_export(force=force):
            return False
        try:
            self._ensure_sheets(tables.keys())
            for sheet_name, df in tables.items():
                self._update_sheet(sheet_name, df)
            self._last_export_monotonic = time.monotonic()
            tprint(
                "Google Sheets trade export complete: "
                + ", ".join(f"{name}={len(df)}" for name, df in tables.items())
            )
            return True
        except Exception as exc:
            tprint(f"Google Sheets trade export failed: {type(exc).__name__}: {exc}")
            return False

    def export_trade_logger(
        self,
        trade_logger: TradeLogger,
        *,
        active_positions: Optional[Dict[str, Dict[str, Any]]] = None,
        force: bool = False,
    ) -> bool:
        logs = trade_logger.read_logs()
        tables = build_google_sheets_trade_tables(
            logs,
            active_positions=active_positions,
        )
        return self.export_tables(tables, force=force)
