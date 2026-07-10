"""
Trade Logger for Inference.

This module handles CSV logging of trade decisions with detailed metrics:
- Log all trade decisions with full context
- Columns for audit trail
- Human-readable trade explanations
"""

import csv
import json
import os
import shutil
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import numpy as np
import pandas as pd

from extreme_price_movements.utils import tprint

# Default log directory
DEFAULT_LOG_DIR = "extreme_price_movements/logs"


def _safe_finite_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _derive_entry_notional_quote(decision: Dict[str, Any]) -> Any:
    """Return a positive quote notional for fee/PnL audit when the direct field is missing."""
    direct_keys = (
        "entry_notional_quote",
        "notional_quote",
        "position_size_after_liquidity",
        "intended_quote_size",
        "position_size_before_liquidity",
        "quote_size",
        "ridge_position_size",
    )
    for key in direct_keys:
        value = _safe_finite_float(decision.get(key))
        if np.isfinite(value) and abs(value) > 0.0:
            return abs(value)

    base_amount = _safe_finite_float(
        decision.get("requested_base_amount") or decision.get("base_amount")
    )
    if np.isfinite(base_amount) and abs(base_amount) > 0.0:
        for key in (
            "realized_entry_price",
            "actual_entry_price",
            "price",
            "expected_entry_price",
            "entry_px",
        ):
            price = _safe_finite_float(decision.get(key))
            if np.isfinite(price) and abs(price) > 0.0:
                return abs(base_amount * price)

    fee_quote = _safe_finite_float(decision.get("entry_fee_estimate_quote"))
    fee_bps = _safe_finite_float(decision.get("entry_fee_estimate_bps"))
    if (
        np.isfinite(fee_quote)
        and abs(fee_quote) > 0.0
        and np.isfinite(fee_bps)
        and abs(fee_bps) > 0.0
    ):
        return abs(fee_quote) * 10000.0 / abs(fee_bps)

    return decision.get("entry_notional_quote")


def _is_missing_log_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == "" or value.strip().lower() in {"nan", "none", "null"}
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _fill_best_available_net_fields(record: Dict[str, Any]) -> Dict[str, Any]:
    """Populate primary net-PnL log fields from estimated net fields when needed."""
    out = dict(record)

    def _copy_if_missing(dst: str, src: str) -> None:
        if _is_missing_log_value(out.get(dst)) and not _is_missing_log_value(
            out.get(src)
        ):
            out[dst] = out.get(src)

    _copy_if_missing("net_pnl", "net_pnl_estimated")
    _copy_if_missing("net_pnl_amount", "net_pnl_estimated")
    _copy_if_missing("net_pnl_pct", "net_pnl_pct_estimated")
    _copy_if_missing("fees_amount", "estimated_fees_amount")
    _copy_if_missing("gross_to_net_cost_quote", "gross_to_estimated_net_cost_quote")
    _copy_if_missing("gross_to_net_cost_pct", "gross_to_estimated_net_cost_pct")
    _copy_if_missing(
        "gross_to_net_friction_drag_bps",
        "gross_to_estimated_net_friction_drag_bps",
    )
    if (
        not _is_missing_log_value(out.get("estimated_fees_amount"))
        and _is_missing_log_value(out.get("fees_estimated"))
    ):
        out["fees_estimated"] = True
    if (
        not _is_missing_log_value(out.get("estimated_fee_source"))
        and _is_missing_log_value(out.get("fee_source"))
    ):
        out["fee_source"] = out.get("estimated_fee_source")
    if (
        not _is_missing_log_value(out.get("net_pnl_pct_estimated"))
        and _is_missing_log_value(out.get("net_pnl_verification_status"))
    ):
        out["net_pnl_verification_status"] = "estimated_missing_exchange_fees"
    return out


# Expanded CSV columns for detailed trade logging
TRADE_LOG_COLUMNS = [
    # Core identifiers
    "timestamp",
    "run_id",
    "decision_ts",
    "signal_bar_ts",
    "feature_source_max_ts",
    "feature_available_ts",
    "feature_contract_hash",
    "model_artifact_run_id",
    "trade_id",
    "position_id",
    "lifecycle_event",
    "symbol",
    "side",
    "action",
    "was_traded",
    "portfolio_decision",
    "portfolio_reject_reason",
    "liquidity_reject_reason",
    "mode",
    "strategy_id",
    # Asset & market context
    "entry_price",
    "expected_entry_price",
    "realized_entry_price",
    "entry_order_type",
    "quote_size",
    "requested_base_amount",
    "entry_time",
    "exit_time",
    "decision_to_entry_seconds",
    "signal_close_to_entry_seconds",
    "signal_to_entry_seconds",
    "entry_notional_quote",
    "exit_notional_quote",
    "holding_time_hours",
    "wallet_value_at_entry",
    "open_notional_at_entry",
    "leverage_wallet_multiplier",
    "effective_position_leverage",
    "gross_pnl_pct_wallet",
    "net_pnl_pct_wallet",
    "leverage_adjusted_gross_pnl_pct",
    "leverage_adjusted_net_pnl_pct",
    "net_pnl_pct_wallet_estimated",
    "leverage_adjusted_net_pnl_pct_estimated",
    "configured_entry_leverage",
    "gross_pnl_pct_configured_leverage",
    "net_pnl_pct_configured_leverage",
    "net_pnl_pct_configured_leverage_estimated",
    "requested_entry_leverage",
    "actual_entry_leverage",
    "exchange_entry_leverage",
    "max_entry_leverage",
    "perp_default_leverage",
    "perp_rank_leverage",
    "perp_legacy_risk_cap_leverage",
    "perp_liquidation_risk_cap_leverage",
    "perp_risk_cap_leverage",
    "perp_effective_leverage",
    "perp_stop_loss_pct",
    "perp_liquidation_guard_enabled",
    "perp_liquidation_guard_reason",
    "perp_liquidation_requested_leverage",
    "perp_liquidation_guarded_leverage",
    "perp_liquidation_safe_max_leverage",
    "perp_liquidation_leverage_capped",
    "perp_liquidation_guard_reject",
    "perp_liquidation_stop_distance_pct",
    "perp_liquidation_stop_distance_bps",
    "perp_liquidation_required_distance_pct",
    "perp_liquidation_distance_at_requested_pct",
    "perp_liquidation_distance_at_guarded_pct",
    "perp_liquidation_maintenance_margin_pct",
    "perp_liquidation_fee_buffer_pct",
    "perp_liquidation_safety_buffer_pct",
    "price_slippage_pct",
    "ohlcv_entry_price",
    "entry_price_delta_vs_ohlcv",
    "entry_price_delta_vs_ohlcv_pct",
    "signal_price",
    "decision_mid",
    "signal_gap_bps",
    "ticker_bid",
    "ticker_ask",
    "ticker_mid",
    "ticker_spread_bps",
    "expected_spread_bps",
    "expected_spread_source",
    "expected_half_spread_bps",
    "entry_spread_bps",
    "entry_spread_source",
    "entry_vs_expected_spread_bps",
    "actual_exit_spread_bps",
    "actual_exit_ticker_spread_bps",
    "actual_exit_orderbook_spread_bps",
    "actual_exit_spread_source",
    "exit_vs_expected_spread_bps",
    "actual_exit_bid",
    "actual_exit_ask",
    "actual_exit_last",
    "close_execution_method",
    "close_execution_detail",
    "close_price_source",
    "close_trigger_type",
    "close_trigger_reference",
    "close_touch_side",
    "sentinel_executable_price",
    "sentinel_executable_price_source",
    "sentinel_stop_distance_bps",
    "sentinel_stop_breach_overshoot_bps",
    "sentinel_pretrigger_enabled",
    "sentinel_pretrigger_buffer_bps",
    "sentinel_pretriggered",
    "last_lightweight_stop_sentinel_ts",
    "expected_fill_price",
    "expected_fill_slippage_bps",
    "orderbook_slippage_bps",
    "slippage_bps",
    "entry_gap_bps",
    "entry_slippage_proxy_bps",
    "adverse_signal_gap_bps",
    "expected_total_entry_friction_bps",
    "expected_friction_drag_bps",
    "ev_haircut_bps",
    "ev_haircut_raw_live_entry_friction_bps",
    "ev_haircut_observed_spread_bps",
    "ev_haircut_observed_half_spread_bps",
    "ev_haircut_spread_baseline_bps",
    "ev_haircut_spread_baseline_source",
    "ev_haircut_half_spread_baseline_bps",
    "ev_haircut_spread_excess_bps",
    "ev_haircut_orderbook_slippage_bps",
    "ev_haircut_adverse_signal_gap_bps",
    "ev_haircut_observed_delay_slippage_bps",
    "ev_haircut_delay_slippage_baseline_bps",
    "ev_haircut_delay_slippage_excess_bps",
    "ev_haircut_expected_stop_exit_friction_bps",
    "ev_haircut_stop_exit_baseline_bps",
    "ev_haircut_stop_exit_excess_bps",
    "ev_haircut_stop_exit_source",
    "ev_haircut_contract",
    "ev_adjusted_entry_friction_bps",
    "ev_adjusted_net_return_before_friction",
    "ev_adjusted_net_return_after_friction",
    "ev_adjusted_calibrated_score",
    "ev_adjusted_rank_score",
    "ev_adjusted_source",
    "entry_delay_effect_bps",
    "entry_delay_adverse_bps",
    "entry_delay_abs_bps",
    "gross_to_net_friction_drag_bps",
    "realized_fee_bps",
    "realized_funding_bps",
    "realized_borrow_bps",
    "orderbook_side",
    "best_touch",
    "max_walk_price",
    "orderbook_capacity_quote_within_slippage",
    "intended_quote_size",
    "spread_weight",
    "depth_weight",
    "liquidity_capacity_weight",
    "price_gap_penalty",
    "adjusted_rank_score",
    "final_threshold",
    "position_size_before_liquidity",
    "position_size_after_liquidity",
    "max_chase_bps",
    "entry_limit_price",
    "spread_proxy_pct",
    "atr",
    "atr_frac",
    "volume",
    "vol_zscore",
    "ret24h",
    "range_12h_pct",
    "volatility_zscore",
    # Model predictions - Alpha (base) models
    "alpha_long_mr_pred",
    "alpha_long_tf_pred",
    "alpha_short_mr_pred",
    "alpha_short_tf_pred",
    # Model predictions - Meta model
    "meta_pred",
    "meta_confidence",
    "calibrated_score",
    "estimated_hit_rate",
    "estimated_hit_rate_source",
    "estimated_hit_rate_calibration_n",
    "estimated_ev_gross_return",
    "estimated_ev_net_return",
    "estimated_ev_cost_bps",
    "estimated_ev_hit_rate",
    "estimated_ev_source",
    "estimated_ev_calibration_n",
    "rank_threshold",
    "rank_percentile",
    "deployment_rank_threshold",
    "policy_archetype",
    "local_side_archetype",
    "policy_archetype_source",
    "archetype_hit_surprise_threshold",
    "archetype_hit_surprise_mode",
    "archetype_hit_surprise_applied",
    "archetype_hit_surprise_reason",
    "archetype_hit_surprise_matched_key",
    "archetype_hit_surprise_threshold_delta",
    "archetype_hit_surprise_quality_adjustment",
    "archetype_hit_surprise_priority_multiplier",
    "archetype_hit_surprise_priority_adjustment",
    "archetype_hit_surprise_rank_adjustment",
    "archetype_hit_surprise_actual_hit_rate",
    "archetype_hit_surprise_expected_hit_rate",
    "archetype_hit_surprise_hit_rate_delta",
    "archetype_hit_surprise_hit_rate_surprise_z",
    "archetype_hit_surprise_support_confidence",
    "archetype_hit_surprise_n_eff",
    "archetype_hit_surprise_rows",
    "strategy_ev_threshold",
    "strategy_ev_threshold_before_dynamic",
    "strategy_ev_threshold_source",
    "strategy_ev_threshold_enabled",
    "strategy_ev_threshold_reason",
    "strategy_ev_threshold_mean_net_return",
    "strategy_ev_threshold_hit_rate",
    "strategy_ev_threshold_n_trades",
    "strategy_ev_gate_allowed",
    "strategy_ev_gate_reason",
    "strategy_ev_avg_net_return",
    "strategy_ev_hit_rate",
    "strategy_ev_target_mean_net_return",
    "strategy_ev_min_hit_rate",
    "strategy_ev_diagnostic_threshold",
    "strategy_ev_diagnostic_threshold_enabled",
    "strategy_ev_diagnostic_threshold_reason",
    "policy_artifact_run_id",
    "policy_schema_version",
    "base_pred",
    "base_rank_pct",
    "base_train_rank_pct",
    "base_gate_top_frac",
    "meta_train_rank_pct",
    "rank_score_source",
    "sizer_rank_percentile",
    "effective_threshold",
    "decision_audit_schema",
    "model_prediction_audit",
    "raw_data_audit",
    "model_feature_audit",
    # Model predictions - Ridge position sizer
    "ridge_position_size",
    "ridge_confidence",
    # Entry policy
    "place_order",
    "eu_star",
    "u_hat_z",
    "mae_hat_z",
    "mfe_hat_z",
    "limit_offset_bps",
    "sl_distance_atr",
    "tp_distance_atr",
    # Regime features (for explaining why)
    "G_VOL",
    "G_TREND",
    "G_VOLUME",
    "vol_z",
    "trend_pct",
    "trend",
    "entropy",
    "vol_of_vol",
    "kurtosis",
    "jump_frequency",
    "mkt_rv_ratio",
    "funding_cost",
    "borrow_cost",
    # Candidate selection thresholds used
    "threshold_extreme_pct",
    "threshold_min_range",
    "threshold_min_vol_zscore",
    # Disagreement features
    "disagree_mr_std",
    "disagree_tf_std",
    "agree_tf_minus_mr",
    # OCO order details (live mode)
    "oco_id",
    "exchange_order_id",
    "stop_price",
    "stop_order_id",
    "policy_stop_price",
    "exchange_stop_price",
    "exchange_stop_trigger_reference_source",
    "exchange_stop_adjustment",
    "stop_trigger_signal",
    "stop_trigger_reference_source",
    "stop_policy_params_source",
    "stop_policy_params_hash",
    "stop_policy_schema",
    "decision_module",
    "shadow_policy_schema",
    "shadow_policy_params_source",
    "shadow_policy_params_hash",
    "shadow_policy_entry_price",
    "shadow_realized_entry_price",
    "shadow_entry_gap_bps",
    "shadow_initial_stop_price",
    "shadow_latest_stop_price",
    "shadow_live_stop_price",
    "shadow_stop_gap_bps",
    "shadow_exit_time",
    "shadow_exit_price",
    "shadow_exit_price_source",
    "shadow_theoretical_exit_price",
    "shadow_stop_trigger_price",
    "shadow_trigger_vs_live_exit_gap_bps",
    "shadow_exit_reason",
    "shadow_exit_return",
    "shadow_status",
    "simple_policy_shadow",
    "stop_price_updated",
    "limit_price",
    "exit_reason",
    "actual_entry_price",
    "actual_exit_price",
    "realized_exit_price",
    # Aggtrades data (live mode)
    "aggtrades_count",
    "orderbook_snapshot",
    "gross_pnl_pct",
    "net_pnl_pct",
    "gross_pnl_amount",
    "net_pnl_amount",
    "fees_amount",
    "entry_fee_quote",
    "exit_fee_quote",
    "fee_source",
    "entry_fee_source",
    "exit_fee_source",
    "fees_verified",
    "entry_fee_estimate_quote",
    "entry_fee_estimate_bps",
    "entry_fee_estimate_source",
    "exit_fee_estimate_quote",
    "exit_fee_estimate_bps",
    "exit_fee_estimate_source",
    "estimated_fees_amount",
    "estimated_fee_source",
    "fees_estimated",
    "fees_estimated_complete",
    "net_pnl_estimated",
    "net_pnl_pct_estimated",
    "gross_to_estimated_net_cost_quote",
    "gross_to_estimated_net_cost_pct",
    "gross_to_estimated_net_friction_drag_bps",
    "net_pnl_verification_status",
    "gross_to_net_cost_quote",
    "gross_to_net_cost_pct",
    "net_pnl",
    "mfe",
    "mae",
    "requested_policy_stop",
    "final_placed_stop",
    "exit_vs_policy_stop_bps",
    "exit_vs_peak_giveback_pct",
    "policy_parity_ok",
    "exit_reason_detail",
    "trade_recap",
    "expected_hit_rate",
    "realized_hit_rate",
    "calibration_error",
    # Status
    "status",
    "order_error_category",
    "error",
]


def _derive_holding_time_hours(record: Dict[str, Any]) -> str:
    """Return canonical holding time hours from explicit or timestamp fields."""
    for key in ("holding_time_hours", "time_in_trade_hours"):
        value = record.get(key)
        if value is None or value == "":
            continue
        try:
            out = float(value)
            if np.isfinite(out):
                return str(out)
        except (TypeError, ValueError):
            continue

    entry_time = record.get("entry_time")
    exit_time = record.get("exit_time") or record.get("timestamp")
    if not entry_time or not exit_time:
        return ""
    entry_ts = pd.to_datetime(entry_time, utc=True, errors="coerce")
    exit_ts = pd.to_datetime(exit_time, utc=True, errors="coerce")
    if pd.isna(entry_ts) or pd.isna(exit_ts):
        return ""
    hours = float(
        (pd.Timestamp(exit_ts) - pd.Timestamp(entry_ts)).total_seconds() / 3600.0
    )
    return str(hours) if np.isfinite(hours) else ""


def _classify_trade_log_error(error: Any) -> str:
    """Return a stable execution-error category for reporting."""
    text = str(error or "").lower()
    if not text:
        return ""
    if "max pledged collateral" in text or "max transfer in quantity is 0" in text:
        return "asset_collateral_limit"
    if "insufficient" in text or "balance" in text or "margin" in text:
        return "insufficient_balance"
    if "precision" in text or "lot_size" in text or "min_notional" in text:
        return "invalid_precision_or_filter"
    if "rate limit" in text or "too many requests" in text or "429" in text:
        return "rate_limited"
    if "timeout" in text or "network" in text:
        return "network_timeout"
    if "permission" in text or "unauthorized" in text or "authentication" in text:
        return "auth_or_permission"
    if "reject" in text or "invalidorder" in text:
        return "order_rejected"
    return "exchange_error"


@dataclass
class TradeLogger:
    """Logs trade decisions to CSV for audit trail with detailed metrics."""

    output_path: str = "inference_trades.csv"
    run_id: Optional[str] = None
    db_path: Optional[str] = None

    # Internal state
    _log_file: str = field(init=False, repr=False)
    _initialized: bool = field(default=False, init=False)

    def __post_init__(self):
        """Initialize the logger after dataclass initialization."""
        self.run_id = self.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

        # Ensure directory exists
        log_dir = os.path.dirname(self.output_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        self._log_file = self.output_path
        self.db_path = self.db_path or os.path.splitext(self.output_path)[0] + ".sqlite"

        self._init_db()
        # SQLite is the canonical store. Keep CSV as a regenerated reporting
        # view so schema additions cannot shift historical rows.
        if self._sqlite_has_rows():
            self._sync_csv_from_db(backup_if_changed=True)
        elif not os.path.exists(self._log_file):
            self._write_header()
        else:
            self._ensure_csv_schema()

        self._initialized = True
        tprint(f"TradeLogger initialized: {self._log_file}")

    @property
    def columns(self) -> List[str]:
        """Return the list of columns for trade logging."""
        return TRADE_LOG_COLUMNS

    def _write_header(self):
        """Write CSV header."""
        with open(self._log_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=TRADE_LOG_COLUMNS)
            writer.writeheader()

    def _ensure_csv_schema(self) -> None:
        """Backfill newly added trade log columns into existing CSV logs."""
        try:
            with open(self._log_file, newline="") as f:
                reader = csv.reader(f)
                header = next(reader, [])
            if list(header) == TRADE_LOG_COLUMNS:
                return
            missing = [col for col in TRADE_LOG_COLUMNS if col not in header]
            if not missing:
                return
            existing = pd.read_csv(self._log_file)
            existing = existing.reindex(columns=TRADE_LOG_COLUMNS, fill_value="")
            existing.to_csv(self._log_file, index=False)
            tprint(f"TradeLogger CSV schema updated with columns: {missing}")
        except Exception as exc:
            tprint(f"TradeLogger CSV schema check failed: {exc}")

    def _init_db(self) -> None:
        """Create the durable trade diagnostics table if needed."""
        if not self.db_path:
            return
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        columns = ", ".join(f'"{col}" TEXT' for col in TRADE_LOG_COLUMNS)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    {columns},
                    record_hash TEXT UNIQUE
                )
                """
            )
            existing = {
                str(row[1])
                for row in conn.execute('PRAGMA table_info("trades")').fetchall()
            }
            for col in TRADE_LOG_COLUMNS:
                if col not in existing:
                    conn.execute(f'ALTER TABLE trades ADD COLUMN "{col}" TEXT')
            self._backfill_missing_lifecycle_ids(conn)
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_trades_trade_id "
                'ON trades("trade_id")'
            )
            conn.commit()

    def _backfill_missing_lifecycle_ids(self, conn: sqlite3.Connection) -> None:
        """Populate lifecycle identifiers for rows logged before they existed."""
        conn.row_factory = sqlite3.Row
        rows = conn.execute('SELECT * FROM "trades" ORDER BY id ASC').fetchall()
        for row in rows:
            record = {
                col: row[col] if col in row.keys() else "" for col in TRADE_LOG_COLUMNS
            }
            normalized = self._normalize_record(record)
            updates: Dict[str, str] = {}
            for col in (
                "trade_id",
                "position_id",
                "lifecycle_event",
                "order_error_category",
            ):
                should_update = not row[col] and normalized.get(col)
                if (
                    col == "order_error_category"
                    and row[col] == "exchange_error"
                    and normalized.get(col)
                    and normalized[col] != "exchange_error"
                ):
                    should_update = True
                if should_update:
                    updates[col] = self._stringify_value(normalized[col])
            if not updates:
                continue
            assignments = ", ".join(f'"{col}" = ?' for col in updates)
            conn.execute(
                f'UPDATE "trades" SET {assignments} WHERE id = ?',
                [*updates.values(), row["id"]],
            )

    def _sqlite_has_rows(self) -> bool:
        """Return True when the canonical sqlite log has any rows."""
        if not self.db_path or not os.path.exists(self.db_path):
            return False
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute('SELECT COUNT(*) FROM "trades"').fetchone()
            return bool(row and int(row[0]) > 0)
        except Exception:
            return False

    def _read_db_logs(self) -> pd.DataFrame:
        """Read canonical sqlite logs into the public CSV column order."""
        if not self.db_path or not os.path.exists(self.db_path):
            return pd.DataFrame(columns=TRADE_LOG_COLUMNS)
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(
                    'SELECT * FROM "trades" ORDER BY id ASC',
                    conn,
                )
        except Exception as exc:
            tprint(f"TradeLogger sqlite read failed: {exc}")
            return pd.DataFrame(columns=TRADE_LOG_COLUMNS)
        if df.empty:
            return pd.DataFrame(columns=TRADE_LOG_COLUMNS)
        return df.reindex(columns=TRADE_LOG_COLUMNS, fill_value="")

    def _sync_csv_from_db(self, *, backup_if_changed: bool = False) -> None:
        """Regenerate the reporting CSV from sqlite."""
        df = self._read_db_logs()
        if df.empty:
            if not os.path.exists(self._log_file):
                self._write_header()
            return
        path = Path(self._log_file)
        if backup_if_changed and path.exists():
            try:
                existing = pd.read_csv(path, dtype=str).reindex(
                    columns=TRADE_LOG_COLUMNS,
                    fill_value="",
                )
                if not existing.fillna("").equals(df.fillna("")):
                    backup = path.with_suffix(path.suffix + ".pre_sqlite_sync.bak")
                    if not backup.exists():
                        shutil.copy2(path, backup)
            except Exception:
                backup = path.with_suffix(path.suffix + ".pre_sqlite_sync.bak")
                if not backup.exists():
                    shutil.copy2(path, backup)
        df.to_csv(self._log_file, index=False)

    def _stringify_value(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, sort_keys=True, default=str)
        return str(value)

    def _default_lifecycle_event(self, record: Dict[str, Any]) -> str:
        """Infer a lifecycle event when callers only pass legacy action/status."""
        explicit = record.get("lifecycle_event")
        if explicit:
            return str(explicit)
        action = str(record.get("action", "") or "").lower()
        status = str(record.get("status", "") or "").lower()
        if status in {"failed", "rejected", "error"}:
            return f"{action or 'trade'}_failed"
        if action == "enter":
            if status in {"pending", "recorded"}:
                return "entry_placed"
            return "entry_recorded"
        if action == "exit":
            if status in {"closed", "completed"}:
                return "exit_filled"
            return "exit_update"
        if "stop" in action:
            return "stop_replaced" if "replace" in action else "stop_update"
        return action or "trade_event"

    def _derive_position_id(self, record: Dict[str, Any]) -> str:
        """Build a stable position id from exchange ids or deterministic context."""
        explicit = record.get("position_id")
        if explicit:
            return self._stringify_value(explicit)
        for key in ("exchange_order_id", "oco_id"):
            value = record.get(key)
            if value:
                return f"{record.get('run_id', self.run_id)}:{value}"
        parts = [
            record.get("run_id", self.run_id),
            record.get("symbol", ""),
            record.get("side", ""),
            record.get("strategy_id", ""),
            record.get("expected_entry_price")
            or record.get("realized_entry_price")
            or record.get("entry_price", ""),
            record.get("timestamp", ""),
        ]
        return "|".join(self._stringify_value(part) for part in parts)

    def _derive_trade_id(self, record: Dict[str, Any]) -> str:
        """Build a unique event id under a position lifecycle."""
        explicit = record.get("trade_id")
        if explicit:
            return self._stringify_value(explicit)
        parts = [
            record.get("position_id") or self._derive_position_id(record),
            record.get("action", ""),
            record.get("lifecycle_event") or self._default_lifecycle_event(record),
            record.get("timestamp", ""),
            record.get("stop_order_id", ""),
            record.get("realized_exit_price", ""),
        ]
        return "|".join(self._stringify_value(part) for part in parts)

    def _normalize_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        enriched = _fill_best_available_net_fields(dict(record))
        enriched["lifecycle_event"] = self._default_lifecycle_event(enriched)
        enriched["position_id"] = self._derive_position_id(enriched)
        enriched["trade_id"] = self._derive_trade_id(enriched)
        holding_time_hours = _derive_holding_time_hours(enriched)
        if holding_time_hours:
            enriched["holding_time_hours"] = holding_time_hours
        if enriched.get("error"):
            classified_error = _classify_trade_log_error(enriched.get("error"))
            if not enriched.get("order_error_category") or (
                enriched.get("order_error_category") == "exchange_error"
                and classified_error
                and classified_error != "exchange_error"
            ):
                enriched["order_error_category"] = classified_error
        return {col: enriched.get(col, "") for col in TRADE_LOG_COLUMNS}

    def _write_db_record(self, record: Dict[str, Any]) -> None:
        """Append or update a trade row in sqlite with idempotent protection."""
        if not self.db_path:
            return
        normalized = self._normalize_record(record)
        record_hash = "|".join(
            self._stringify_value(normalized.get(col))
            for col in (
                "timestamp",
                "run_id",
                "symbol",
                "side",
                "action",
                "strategy_id",
                "expected_entry_price",
                "realized_entry_price",
            )
        )
        cols = list(TRADE_LOG_COLUMNS) + ["record_hash"]
        placeholders = ", ".join("?" for _ in cols)
        quoted_cols = ", ".join(f'"{col}"' for col in cols)
        update_cols = [col for col in TRADE_LOG_COLUMNS if col != "trade_id"]
        update_clause = ", ".join(
            f'"{col}" = CASE '
            f'WHEN excluded."{col}" != "" THEN excluded."{col}" '
            f'ELSE "trades"."{col}" END'
            for col in update_cols
        )
        values = [
            self._stringify_value(normalized.get(col)) for col in TRADE_LOG_COLUMNS
        ]
        values.append(record_hash)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                f"""
                INSERT INTO trades ({quoted_cols}) VALUES ({placeholders})
                ON CONFLICT(trade_id) DO UPDATE SET
                    {update_clause},
                    record_hash = excluded.record_hash
                """,
                values,
            )
            conn.commit()

    def _persist_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Persist record to sqlite and refresh the CSV reporting view."""
        normalized = self._normalize_record(record)
        self._write_db_record(normalized)
        self._sync_csv_from_db()
        return normalized

    def log_trade(
        self,
        decision: Dict[str, Any],
        model_results: Dict[str, Any],
        market_data: Dict[str, Any],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Log a trade decision with full context.

        Args:
            decision: Output from model_orchestrator.run_full_chain()
            model_results: All model predictions and intermediate results
            market_data: Current market data (price, volume, ATR, etc.)
            config: Config used for this inference run

        Returns:
            The record that was written to the CSV
        """
        # Extract nested values with defaults
        alpha_preds = model_results.get("alpha_preds", {})
        entry_policy = decision.get("entry_policy", {})
        disagreement_features = model_results.get("disagreement_features", {})

        record = {
            # Core identifiers
            "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
            "run_id": config.get("run_id", self.run_id),
            "trade_id": decision.get("trade_id"),
            "position_id": decision.get("position_id"),
            "lifecycle_event": decision.get("lifecycle_event"),
            "symbol": decision.get("symbol"),
            "side": decision.get("side"),
            "action": decision.get("action"),
            "mode": config.get("mode", "shadow"),
            "strategy_id": decision.get("strategy_id"),
            # Asset & market context
            "entry_price": market_data.get("close"),
            "expected_entry_price": decision.get("expected_entry_price")
            or decision.get("entry_px"),
            "realized_entry_price": decision.get("realized_entry_price")
            or decision.get("price"),
            "entry_order_type": decision.get("entry_order_type"),
            "quote_size": decision.get("quote_size") or decision.get("size"),
            "requested_base_amount": decision.get("requested_base_amount")
            or decision.get("base_amount"),
            "entry_time": decision.get("entry_time"),
            "exit_time": decision.get("exit_time"),
            "decision_to_entry_seconds": decision.get("decision_to_entry_seconds"),
            "signal_to_entry_seconds": decision.get("signal_to_entry_seconds"),
            "entry_notional_quote": _derive_entry_notional_quote(decision),
            "exit_notional_quote": decision.get("exit_notional_quote"),
            "holding_time_hours": decision.get("holding_time_hours")
            or decision.get("time_in_trade_hours"),
            "price_slippage_pct": decision.get("price_slippage_pct"),
            "ohlcv_entry_price": decision.get("ohlcv_entry_price"),
            "entry_price_delta_vs_ohlcv": decision.get("entry_price_delta_vs_ohlcv"),
            "entry_price_delta_vs_ohlcv_pct": decision.get(
                "entry_price_delta_vs_ohlcv_pct"
            ),
            "signal_price": decision.get("signal_price"),
            "decision_mid": decision.get("decision_mid"),
            "signal_gap_bps": decision.get("signal_gap_bps"),
            "ticker_bid": decision.get("ticker_bid") or decision.get("bid"),
            "ticker_ask": decision.get("ticker_ask") or decision.get("ask"),
            "ticker_mid": decision.get("ticker_mid") or decision.get("mid"),
            "ticker_spread_bps": decision.get("ticker_spread_bps")
            or decision.get("spread_bps"),
            "expected_spread_bps": decision.get("expected_spread_bps"),
            "expected_spread_source": decision.get("expected_spread_source"),
            "expected_half_spread_bps": decision.get("expected_half_spread_bps"),
            "entry_spread_bps": decision.get("entry_spread_bps"),
            "entry_spread_source": decision.get("entry_spread_source"),
            "entry_vs_expected_spread_bps": decision.get(
                "entry_vs_expected_spread_bps"
            ),
            "actual_exit_spread_bps": decision.get("actual_exit_spread_bps"),
            "actual_exit_ticker_spread_bps": decision.get(
                "actual_exit_ticker_spread_bps"
            ),
            "actual_exit_orderbook_spread_bps": decision.get(
                "actual_exit_orderbook_spread_bps"
            ),
            "actual_exit_spread_source": decision.get("actual_exit_spread_source"),
            "exit_vs_expected_spread_bps": decision.get("exit_vs_expected_spread_bps"),
            "actual_exit_bid": decision.get("actual_exit_bid"),
            "actual_exit_ask": decision.get("actual_exit_ask"),
            "actual_exit_last": decision.get("actual_exit_last"),
            "close_execution_method": decision.get("close_execution_method"),
            "close_execution_detail": decision.get("close_execution_detail"),
            "close_price_source": decision.get("close_price_source"),
            "close_trigger_type": decision.get("close_trigger_type"),
            "close_trigger_reference": decision.get("close_trigger_reference"),
            "close_touch_side": decision.get("close_touch_side"),
            "expected_fill_price": decision.get("expected_fill_price"),
            "expected_fill_slippage_bps": decision.get("expected_fill_slippage_bps"),
            "expected_total_entry_friction_bps": decision.get(
                "expected_total_entry_friction_bps"
            ),
            "expected_friction_drag_bps": decision.get("expected_friction_drag_bps")
            or decision.get("expected_total_entry_friction_bps"),
            "entry_delay_effect_bps": decision.get("entry_delay_effect_bps"),
            "entry_delay_adverse_bps": decision.get("entry_delay_adverse_bps"),
            "entry_delay_abs_bps": decision.get("entry_delay_abs_bps"),
            "gross_to_net_friction_drag_bps": decision.get(
                "gross_to_net_friction_drag_bps"
            ),
            "orderbook_side": decision.get("orderbook_side"),
            "best_touch": decision.get("best_touch"),
            "max_walk_price": decision.get("max_walk_price"),
            "orderbook_capacity_quote_within_slippage": decision.get(
                "orderbook_capacity_quote_within_slippage"
            ),
            "intended_quote_size": decision.get("intended_quote_size"),
            "spread_weight": decision.get("spread_weight"),
            "depth_weight": decision.get("depth_weight"),
            "liquidity_capacity_weight": decision.get("liquidity_capacity_weight"),
            "price_gap_penalty": decision.get("price_gap_penalty"),
            "adjusted_rank_score": decision.get("adjusted_rank_score"),
            "final_threshold": decision.get("final_threshold"),
            "position_size_before_liquidity": decision.get(
                "position_size_before_liquidity"
            ),
            "position_size_after_liquidity": decision.get(
                "position_size_after_liquidity"
            ),
            "max_chase_bps": decision.get("max_chase_bps"),
            "entry_limit_price": decision.get("entry_limit_price"),
            "spread_proxy_pct": decision.get("spread_proxy_pct"),
            "atr": market_data.get("atr"),
            "atr_frac": market_data.get("atr_frac"),
            "volume": market_data.get("volume"),
            "vol_zscore": market_data.get("vol_zscore"),
            "ret24h": market_data.get("ret24h"),
            "range_12h_pct": market_data.get("range_12h_pct"),
            "volatility_zscore": market_data.get("volatility_zscore"),
            # Alpha model predictions
            "alpha_long_mr_pred": alpha_preds.get("long_mr"),
            "alpha_long_tf_pred": alpha_preds.get("long_tf"),
            "alpha_short_mr_pred": alpha_preds.get("short_mr"),
            "alpha_short_tf_pred": alpha_preds.get("short_tf"),
            # Meta model predictions
            "meta_pred": model_results.get("meta_pred"),
            "meta_confidence": model_results.get("meta_confidence"),
            "calibrated_score": decision.get("calibrated_score"),
            "estimated_hit_rate": decision.get("estimated_hit_rate")
            or model_results.get("estimated_hit_rate"),
            "estimated_hit_rate_source": decision.get("estimated_hit_rate_source")
            or model_results.get("estimated_hit_rate_source"),
            "estimated_hit_rate_calibration_n": decision.get(
                "estimated_hit_rate_calibration_n"
            )
            or model_results.get("estimated_hit_rate_calibration_n"),
            "estimated_ev_gross_return": decision.get("estimated_ev_gross_return")
            or model_results.get("estimated_ev_gross_return"),
            "estimated_ev_net_return": decision.get("estimated_ev_net_return")
            or model_results.get("estimated_ev_net_return"),
            "estimated_ev_cost_bps": decision.get("estimated_ev_cost_bps")
            or model_results.get("estimated_ev_cost_bps"),
            "estimated_ev_hit_rate": decision.get("estimated_ev_hit_rate")
            or model_results.get("estimated_ev_hit_rate"),
            "estimated_ev_source": decision.get("estimated_ev_source")
            or model_results.get("estimated_ev_source"),
            "estimated_ev_calibration_n": decision.get("estimated_ev_calibration_n")
            or model_results.get("estimated_ev_calibration_n"),
            "rank_threshold": decision.get("rank_threshold"),
            "rank_percentile": decision.get("rank_percentile")
            or decision.get("sizer_rank_percentile"),
            "deployment_rank_threshold": decision.get("deployment_rank_threshold")
            or decision.get("effective_threshold"),
            "policy_artifact_run_id": decision.get("policy_artifact_run_id")
            or config.get("policy_artifact_run_id"),
            "policy_schema_version": decision.get("policy_schema_version")
            or config.get("policy_schema_version"),
            "base_pred": model_results.get("base_pred") or decision.get("base_pred"),
            "base_rank_pct": model_results.get("base_rank_pct")
            or decision.get("base_rank_pct"),
            "base_train_rank_pct": model_results.get("base_train_rank_pct")
            or decision.get("base_train_rank_pct"),
            "base_gate_top_frac": model_results.get("base_gate_top_frac")
            or decision.get("base_gate_top_frac"),
            "meta_train_rank_pct": model_results.get("meta_train_rank_pct")
            or decision.get("meta_train_rank_pct"),
            "rank_score_source": model_results.get("rank_score_source")
            or decision.get("rank_score_source"),
            "sizer_rank_percentile": decision.get("sizer_rank_percentile"),
            "effective_threshold": decision.get("effective_threshold"),
            # Ridge position sizer
            "ridge_position_size": model_results.get("position_size"),
            "ridge_confidence": model_results.get("ridge_confidence"),
            # Entry policy
            "place_order": entry_policy.get("place_order"),
            "eu_star": entry_policy.get("eu_star"),
            "u_hat_z": entry_policy.get("u_hat_z"),
            "mae_hat_z": entry_policy.get("mae_hat_z"),
            "mfe_hat_z": entry_policy.get("mfe_hat_z"),
            "limit_offset_bps": entry_policy.get("limit_offset_bps_dynamic"),
            "sl_distance_atr": entry_policy.get("sl_distance_atr_eff"),
            "tp_distance_atr": entry_policy.get("tp_distance_atr_eff"),
            # Regime features
            "G_VOL": market_data.get("G_VOL"),
            "G_TREND": market_data.get("G_TREND"),
            "G_VOLUME": market_data.get("G_VOLUME"),
            "vol_z": market_data.get("vol_z"),
            "trend_pct": market_data.get("trend_pct"),
            "trend": market_data.get("trend"),
            "entropy": market_data.get("entropy"),
            "vol_of_vol": market_data.get("vol_of_vol"),
            "kurtosis": market_data.get("kurtosis"),
            "jump_frequency": market_data.get("jump_frequency"),
            "mkt_rv_ratio": market_data.get("mkt_rv_ratio"),
            "funding_cost": market_data.get("funding_cost"),
            "borrow_cost": market_data.get("borrow_cost"),
            # Candidate thresholds
            "threshold_extreme_pct": config.get("extreme_pct"),
            "threshold_min_range": config.get("min_range_pct"),
            "threshold_min_vol_zscore": config.get("min_vol_zscore"),
            # Disagreement features
            "disagree_mr_std": disagreement_features.get("disagree_mr_std"),
            "disagree_tf_std": disagreement_features.get("disagree_tf_std"),
            "agree_tf_minus_mr": disagreement_features.get("agree_tf_minus_mr_avg"),
            # OCO (live mode)
            "oco_id": decision.get("oco_id"),
            "exchange_order_id": decision.get("exchange_order_id"),
            "stop_price": decision.get("stop_price"),
            "stop_order_id": decision.get("stop_order_id"),
            "policy_stop_price": decision.get("policy_stop_price"),
            "exchange_stop_price": decision.get("exchange_stop_price"),
            "exchange_stop_trigger_reference_source": decision.get(
                "exchange_stop_trigger_reference_source"
            ),
            "exchange_stop_adjustment": decision.get("exchange_stop_adjustment"),
            "stop_trigger_signal": decision.get("stop_trigger_signal"),
            "stop_trigger_reference_source": decision.get(
                "stop_trigger_reference_source"
            ),
            "stop_policy_params_source": decision.get("stop_policy_params_source"),
            "stop_policy_params_hash": decision.get("stop_policy_params_hash"),
            "stop_policy_schema": decision.get("stop_policy_schema"),
            "decision_module": decision.get("decision_module"),
            "stop_price_updated": decision.get("stop_price_updated"),
            "limit_price": decision.get("limit_price"),
            "exit_reason": decision.get("exit_reason"),
            "actual_entry_price": decision.get("actual_entry_price")
            or decision.get("realized_entry_price")
            or decision.get("price"),
            "actual_exit_price": decision.get("actual_exit_price"),
            "realized_exit_price": decision.get("realized_exit_price")
            or decision.get("actual_exit_price"),
            # Aggtrades (live mode)
            "aggtrades_count": (
                len(decision.get("aggtrades", [])) if decision.get("aggtrades") else 0
            ),
            "orderbook_snapshot": decision.get("orderbook_snapshot"),
            "gross_pnl_pct": decision.get("gross_pnl_pct"),
            "net_pnl_pct": decision.get("net_pnl_pct"),
            "gross_pnl_amount": decision.get("gross_pnl_amount"),
            "net_pnl_amount": decision.get("net_pnl_amount"),
            "wallet_value_at_entry": decision.get("wallet_value_at_entry"),
            "open_notional_at_entry": decision.get("open_notional_at_entry"),
            "leverage_wallet_multiplier": decision.get("leverage_wallet_multiplier"),
            "configured_entry_leverage": decision.get("configured_entry_leverage"),
            "effective_position_leverage": decision.get("effective_position_leverage"),
            "gross_pnl_pct_wallet": decision.get("gross_pnl_pct_wallet"),
            "net_pnl_pct_wallet": decision.get("net_pnl_pct_wallet"),
            "leverage_adjusted_gross_pnl_pct": decision.get(
                "leverage_adjusted_gross_pnl_pct"
            ),
            "leverage_adjusted_net_pnl_pct": decision.get(
                "leverage_adjusted_net_pnl_pct"
            ),
            "net_pnl_pct_wallet_estimated": decision.get(
                "net_pnl_pct_wallet_estimated"
            ),
            "leverage_adjusted_net_pnl_pct_estimated": decision.get(
                "leverage_adjusted_net_pnl_pct_estimated"
            ),
            "gross_pnl_pct_configured_leverage": decision.get(
                "gross_pnl_pct_configured_leverage"
            ),
            "net_pnl_pct_configured_leverage": decision.get(
                "net_pnl_pct_configured_leverage"
            ),
            "net_pnl_pct_configured_leverage_estimated": decision.get(
                "net_pnl_pct_configured_leverage_estimated"
            ),
            "fees_amount": decision.get("fees_amount"),
            "entry_fee_quote": decision.get("entry_fee_quote"),
            "exit_fee_quote": decision.get("exit_fee_quote"),
            "fee_source": decision.get("fee_source"),
            "entry_fee_source": decision.get("entry_fee_source"),
            "exit_fee_source": decision.get("exit_fee_source"),
            "fees_verified": decision.get("fees_verified"),
            "entry_fee_estimate_quote": decision.get("entry_fee_estimate_quote"),
            "entry_fee_estimate_bps": decision.get("entry_fee_estimate_bps"),
            "entry_fee_estimate_source": decision.get("entry_fee_estimate_source"),
            "exit_fee_estimate_quote": decision.get("exit_fee_estimate_quote"),
            "exit_fee_estimate_bps": decision.get("exit_fee_estimate_bps"),
            "exit_fee_estimate_source": decision.get("exit_fee_estimate_source"),
            "estimated_fees_amount": decision.get("estimated_fees_amount"),
            "estimated_fee_source": decision.get("estimated_fee_source"),
            "fees_estimated": decision.get("fees_estimated"),
            "fees_estimated_complete": decision.get("fees_estimated_complete"),
            "net_pnl_estimated": decision.get("net_pnl_estimated"),
            "net_pnl_pct_estimated": decision.get("net_pnl_pct_estimated"),
            "gross_to_estimated_net_cost_quote": decision.get(
                "gross_to_estimated_net_cost_quote"
            ),
            "gross_to_estimated_net_cost_pct": decision.get(
                "gross_to_estimated_net_cost_pct"
            ),
            "gross_to_estimated_net_friction_drag_bps": decision.get(
                "gross_to_estimated_net_friction_drag_bps"
            ),
            "net_pnl_verification_status": decision.get(
                "net_pnl_verification_status"
            ),
            "gross_to_net_cost_quote": decision.get("gross_to_net_cost_quote"),
            "gross_to_net_cost_pct": decision.get("gross_to_net_cost_pct"),
            "net_pnl": decision.get("net_pnl"),
            "mfe": decision.get("mfe"),
            "mae": decision.get("mae"),
            "requested_policy_stop": decision.get("requested_policy_stop"),
            "final_placed_stop": decision.get("final_placed_stop"),
            "exit_vs_policy_stop_bps": decision.get("exit_vs_policy_stop_bps"),
            "exit_vs_peak_giveback_pct": decision.get("exit_vs_peak_giveback_pct"),
            "policy_parity_ok": decision.get("policy_parity_ok"),
            "exit_reason_detail": decision.get("exit_reason_detail"),
            "trade_recap": decision.get("trade_recap"),
            "expected_hit_rate": decision.get("expected_hit_rate"),
            "realized_hit_rate": decision.get("realized_hit_rate"),
            "calibration_error": decision.get("calibration_error"),
            # Status
            "status": decision.get("status", "completed"),
            "order_error_category": decision.get("order_error_category", ""),
            "error": decision.get("error", ""),
        }

        record = self._persist_record(record)

        # Generate and print explanation
        explanation = self.explain_trade(record)
        tprint(
            f"Logged trade: {record['action']} {record['side']} {record['symbol']} @ {record['entry_price']}"
        )

        return record

    def explain_trade(self, record: Dict[str, Any]) -> str:
        """
        Generate a human-readable explanation of why a trade was taken.

        Args:
            record: Trade record dictionary

        Returns:
            Human-readable explanation string
        """
        lines = []
        lines.append("=" * 60)
        lines.append(
            f"TRADE EXPLANATION: {record['action'].upper()} {record['side'].upper()}"
        )
        lines.append("=" * 60)

        # Core info
        lines.append(f"\n📊 Symbol: {record['symbol']}")
        lines.append(f"   Entry Price: ${record['entry_price']}")
        lines.append(f"   Mode: {record['mode']}")

        # Market context
        lines.append(f"\n📈 Market Context:")
        lines.append(
            f"   24h Return: {record.get('ret24h', 'N/A'):.2%}"
            if record.get("ret24h")
            else "   24h Return: N/A"
        )
        lines.append(
            f"   12h Range: {record.get('range_12h_pct', 'N/A'):.2%}"
            if record.get("range_12h_pct")
            else "   12h Range: N/A"
        )
        lines.append(
            f"   Volatility Z-Score: {record.get('volatility_zscore', 'N/A'):.2f}"
            if record.get("volatility_zscore")
            else "   Volatility Z-Score: N/A"
        )
        lines.append(
            f"   Volume Z-Score: {record.get('vol_zscore', 'N/A'):.2f}"
            if record.get("vol_zscore")
            else "   Volume Z-Score: N/A"
        )
        lines.append(
            f"   ATR: {record.get('atr', 'N/A'):.4f}"
            if record.get("atr")
            else "   ATR: N/A"
        )

        # Model predictions
        lines.append(f"\n🤖 Model Predictions:")

        if record["side"] == "long":
            lines.append(
                f"   Alpha Long MR:  {record.get('alpha_long_mr_pred', 'N/A'):.4f}"
                if record.get("alpha_long_mr_pred") is not None
                else "   Alpha Long MR:  N/A"
            )
            lines.append(
                f"   Alpha Long TF:  {record.get('alpha_long_tf_pred', 'N/A'):.4f}"
                if record.get("alpha_long_tf_pred") is not None
                else "   Alpha Long TF:  N/A"
            )
        else:
            lines.append(
                f"   Alpha Short MR: {record.get('alpha_short_mr_pred', 'N/A'):.4f}"
                if record.get("alpha_short_mr_pred") is not None
                else "   Alpha Short MR: N/A"
            )
            lines.append(
                f"   Alpha Short TF: {record.get('alpha_short_tf_pred', 'N/A'):.4f}"
                if record.get("alpha_short_tf_pred") is not None
                else "   Alpha Short TF: N/A"
            )

        lines.append(
            f"   Meta Prediction: {record.get('meta_pred', 'N/A'):.4f}"
            if record.get("meta_pred") is not None
            else "   Meta Prediction: N/A"
        )
        lines.append(
            f"   Meta Confidence: {record.get('meta_confidence', 'N/A'):.2%}"
            if record.get("meta_confidence")
            else "   Meta Confidence: N/A"
        )

        # Position sizing
        lines.append(f"\n💰 Position Sizing:")
        lines.append(
            f"   Ridge Position Size: {record.get('ridge_position_size', 'N/A'):.4f}"
            if record.get("ridge_position_size")
            else "   Ridge Position Size: N/A"
        )
        lines.append(
            f"   Ridge Confidence: {record.get('ridge_confidence', 'N/A'):.2%}"
            if record.get("ridge_confidence")
            else "   Ridge Confidence: N/A"
        )

        # Entry policy
        lines.append(f"\n🎯 Entry Policy:")
        place_order = record.get("place_order", False)
        lines.append(f"   Place Order: {'✅ YES' if place_order else '❌ NO'}")
        if place_order:
            lines.append(
                f"   EU* (Expected Utility): {record.get('eu_star', 'N/A'):.4f}"
                if record.get("eu_star") is not None
                else "   EU*: N/A"
            )
            lines.append(
                f"   ũ (Predicted Return): {record.get('u_hat_z', 'N/A'):.4f}"
                if record.get("u_hat_z") is not None
                else "   ũ: N/A"
            )
            lines.append(
                f"   MAÊ (Max Adverse): {record.get('mae_hat_z', 'N/A'):.4f}"
                if record.get("mae_hat_z") is not None
                else "   MAÊ: N/A"
            )
            lines.append(
                f"   MFÊ (Max Favorable): {record.get('mfe_hat_z', 'N/A'):.4f}"
                if record.get("mfe_hat_z") is not None
                else "   MFÊ: N/A"
            )
            lines.append(
                f"   Limit Offset (bps): {record.get('limit_offset_bps', 'N/A')}"
                if record.get("limit_offset_bps")
                else "   Limit Offset: N/A"
            )
            lines.append(
                f"   SL Distance (ATR): {record.get('sl_distance_atr', 'N/A'):.2f}"
                if record.get("sl_distance_atr")
                else "   SL Distance: N/A"
            )
            lines.append(
                f"   TP Distance (ATR): {record.get('tp_distance_atr', 'N/A'):.2f}"
                if record.get("tp_distance_atr")
                else "   TP Distance: N/A"
            )

        # Regime context
        lines.append(f"\n🔄 Regime Features:")
        g_vol = record.get("G_VOL", "N/A")
        g_trend = record.get("G_TREND", "N/A")
        g_volume = record.get("G_VOLUME", "N/A")
        lines.append(f"   Volatility Regime: {g_vol}")
        lines.append(f"   Trend Regime: {g_trend}")
        lines.append(f"   Liquidity Regime: {g_volume}")
        lines.append(
            f"   Vol Z-Score: {record.get('vol_z', 'N/A'):.2f}"
            if record.get("vol_z") is not None
            else "   Vol Z-Score: N/A"
        )
        lines.append(
            f"   Trend %: {record.get('trend_pct', 'N/A'):.2%}"
            if record.get("trend_pct") is not None
            else "   Trend %: N/A"
        )

        # Disagreement features
        lines.append(f"\n⚖️ Model Disagreement:")
        lines.append(
            f"   Disagree MR Std: {record.get('disagree_mr_std', 'N/A'):.4f}"
            if record.get("disagree_mr_std") is not None
            else "   Disagree MR Std: N/A"
        )
        lines.append(
            f"   Disagree TF Std: {record.get('disagree_tf_std', 'N/A'):.4f}"
            if record.get("disagree_tf_std") is not None
            else "   Disagree TF Std: N/A"
        )
        lines.append(
            f"   Agree TF - MR: {record.get('agree_tf_minus_mr', 'N/A'):.4f}"
            if record.get("agree_tf_minus_mr") is not None
            else "   Agree TF - MR: N/A"
        )

        # Why trade was taken
        lines.append(f"\n💡 WHY THIS TRADE:")
        if place_order:
            reasons = []

            # Check meta prediction
            meta_pred = record.get("meta_pred")
            if meta_pred is not None:
                if record["side"] == "long" and meta_pred > 0.5:
                    reasons.append("Strong long signal from meta model")
                elif record["side"] == "short" and meta_pred > 0.5:
                    reasons.append("Strong short signal from meta model")

            # Check regime alignment
            if record.get("G_VOL") == "HIGH":
                reasons.append("Trading in high volatility regime (favorable)")
            elif record.get("G_VOL") == "LOW":
                reasons.append("Trading in low volatility regime")

            # Check trend alignment
            trend_pct = record.get("trend_pct")
            if trend_pct is not None:
                if record["side"] == "long" and trend_pct > 0:
                    reasons.append("Long aligned with positive trend")
                elif record["side"] == "short" and trend_pct < 0:
                    reasons.append("Short aligned with negative trend")

            # Check expected utility
            eu_star = record.get("eu_star")
            if eu_star is not None and eu_star > 0:
                reasons.append(f"Positive expected utility (EU*={eu_star:.4f})")

            # Check disagreement
            disagree_mr = record.get("disagree_mr_std")
            if disagree_mr is not None and disagree_mr < 0.1:
                reasons.append("Low disagreement among MR models (high confidence)")

            if reasons:
                for reason in reasons:
                    lines.append(f"   • {reason}")
            else:
                lines.append("   • Entry policy conditions met")
        else:
            lines.append("   • Entry policy conditions NOT met")
            eu_star = record.get("eu_star")
            if eu_star is not None:
                lines.append(f"   • EU* ({eu_star:.4f}) below threshold")

        lines.append("=" * 60)

        return "\n".join(lines)

    def get_log_path(self) -> str:
        """Get the path to the log file."""
        return self._log_file

    def read_logs(self) -> pd.DataFrame:
        """Read trade logs into DataFrame.

        Returns:
            DataFrame of trade logs
        """
        db_logs = self._read_db_logs()
        if not db_logs.empty:
            return db_logs
        if not os.path.exists(self._log_file):
            return pd.DataFrame(columns=TRADE_LOG_COLUMNS)
        return pd.read_csv(self._log_file).reindex(
            columns=TRADE_LOG_COLUMNS,
            fill_value="",
        )

    def get_last_trade_timestamp(self, symbol: str) -> Optional[pd.Timestamp]:
        """Return the latest logged trade timestamp for a symbol."""
        df = self.read_logs()
        if df.empty or "symbol" not in df.columns or "timestamp" not in df.columns:
            return None
        sym_df = df[df["symbol"] == symbol]
        if sym_df.empty:
            return None
        ts = pd.to_datetime(sym_df["timestamp"], utc=True, errors="coerce").dropna()
        if ts.empty:
            return None
        return pd.Timestamp(ts.max())

    def get_last_losing_trade_timestamp(self, symbol: str) -> Optional[pd.Timestamp]:
        """Return the latest closed-trade timestamp with negative realized PnL."""
        df = self.read_logs()
        required = {"symbol", "timestamp"}
        if df.empty or not required.issubset(set(df.columns)):
            return None
        sym_df = df[df["symbol"] == symbol].copy()
        if sym_df.empty:
            return None

        closed_mask = pd.Series(False, index=sym_df.index)
        if "status" in sym_df.columns:
            status = sym_df["status"].astype(str).str.lower()
            closed_mask |= status.isin({"closed", "completed"})
        if "lifecycle_event" in sym_df.columns:
            event = sym_df["lifecycle_event"].astype(str).str.lower()
            closed_mask |= event.str.contains("close|closed|exit", regex=True)

        pnl = pd.Series(np.nan, index=sym_df.index, dtype="float64")
        for col in ("net_pnl_pct", "gross_pnl_pct", "net_pnl", "gross_pnl_amount"):
            if col in sym_df.columns:
                vals = pd.to_numeric(sym_df[col], errors="coerce")
                pnl = pnl.where(pnl.notna(), vals)

        losing_df = sym_df[closed_mask & (pnl < 0.0)]
        if losing_df.empty:
            return None
        ts = pd.to_datetime(losing_df["timestamp"], utc=True, errors="coerce").dropna()
        if ts.empty:
            return None
        return pd.Timestamp(ts.max())

    def reconcile_pending_entries_absent(
        self,
        active_symbols: Iterable[str],
        *,
        reason: str = "absent_from_executor_startup_reconciliation",
    ) -> int:
        """Mark pending entry rows absent when startup reconciliation finds no position."""
        if not self.db_path or not os.path.exists(self.db_path):
            return 0
        active: Set[str] = {str(sym) for sym in active_symbols if str(sym)}
        now = pd.Timestamp.now(tz="UTC").isoformat()
        updated = 0
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT id, symbol
                FROM trades
                WHERE action = 'enter'
                  AND lifecycle_event = 'entry_placed'
                  AND status = 'pending'
                """
            ).fetchall()
            for row in rows:
                symbol = str(row["symbol"] or "")
                if symbol in active:
                    continue
                conn.execute(
                    """
                    UPDATE trades
                    SET status = ?,
                        order_error_category = ?,
                        error = ?,
                        stop_price_updated = ?
                    WHERE id = ?
                    """,
                    (
                        "reconciled_absent",
                        "position_reconciled_absent",
                        reason,
                        now,
                        row["id"],
                    ),
                )
                updated += 1
            conn.commit()
        if updated:
            self._sync_csv_from_db()
            tprint(
                "TradeLogger reconciled pending entries absent from exchange state: "
                f"updated={updated}"
            )
        return updated

    # =========================================================================
    # Legacy methods for backward compatibility
    # =========================================================================

    def log_trade_legacy(
        self,
        symbol: str,
        side: str,
        action: str,
        size: float,
        price: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None,
        mode: str = "shadow",
        status: str = "pending",
        error: Optional[str] = None,
    ):
        """Legacy method for logging trade decisions.

        Args:
            symbol: Trading symbol
            side: "long" or "short"
            action: "enter" or "exit"
            size: Position size
            price: Entry/exit price
            context: Additional context (predictions, features, etc.)
            mode: "live" or "shadow"
            status: Trade status
            error: Error message if any
        """
        row = {
            "timestamp": datetime.now().isoformat(),
            "run_id": self.run_id,
            "symbol": symbol,
            "side": side,
            "action": action,
            "entry_price": price,
            "ridge_position_size": size,
            "mode": mode,
            "status": status,
            "error": error or "",
        }

        # Add context fields (map to new columns)
        if context:
            row["ridge_position_size"] = context.get(
                "position_size", row.get("ridge_position_size", "")
            )
            row["trade_id"] = context.get("trade_id", "")
            row["position_id"] = context.get("position_id", "")
            row["lifecycle_event"] = context.get("lifecycle_event", "")
            row["meta_pred"] = context.get(
                "meta_pred",
                context.get("meta_mr_pred", ""),
            )
            row["alpha_long_mr_pred"] = context.get("alpha_mr_pred", "")
            row["alpha_long_tf_pred"] = context.get("alpha_tf_pred", "")
            row["disagree_mr_std"] = context.get("disagreement_mr", "")
            row["disagree_tf_std"] = context.get("disagreement_tf", "")
            row["ret24h"] = context.get("ret24h", "")
            row["range_12h_pct"] = context.get("range_12h_pct", "")
            row["volatility_zscore"] = context.get("volatility_zscore", "")
            row["strategy_id"] = context.get("strategy_id", "")
            row["calibrated_score"] = context.get("calibrated_score", "")
            row["estimated_hit_rate"] = context.get("estimated_hit_rate", "")
            row["estimated_hit_rate_source"] = context.get(
                "estimated_hit_rate_source", ""
            )
            row["estimated_hit_rate_calibration_n"] = context.get(
                "estimated_hit_rate_calibration_n", ""
            )
            row["rank_threshold"] = context.get("rank_threshold", "")
            row["rank_percentile"] = context.get(
                "rank_percentile",
                context.get("sizer_rank_percentile", ""),
            )
            row["deployment_rank_threshold"] = context.get(
                "deployment_rank_threshold",
                context.get("effective_threshold", ""),
            )
            row["base_train_rank_pct"] = context.get("base_train_rank_pct", "")
            row["meta_train_rank_pct"] = context.get("meta_train_rank_pct", "")
            row["rank_score_source"] = context.get("rank_score_source", "")
            row["policy_artifact_run_id"] = context.get("policy_artifact_run_id", "")
            row["policy_schema_version"] = context.get("policy_schema_version", "")
            row["expected_entry_price"] = context.get("expected_entry_price", "")
            row["realized_entry_price"] = context.get("realized_entry_price", "")
            row["entry_order_type"] = context.get("entry_order_type", "")
            row["price_slippage_pct"] = context.get("price_slippage_pct", "")
            row["ohlcv_entry_price"] = context.get("ohlcv_entry_price", "")
            row["entry_price_delta_vs_ohlcv"] = context.get(
                "entry_price_delta_vs_ohlcv", ""
            )
            row["entry_price_delta_vs_ohlcv_pct"] = context.get(
                "entry_price_delta_vs_ohlcv_pct", ""
            )
            row["signal_price"] = context.get("signal_price", "")
            row["decision_mid"] = context.get("decision_mid", "")
            row["signal_gap_bps"] = context.get("signal_gap_bps", "")
            row["ticker_bid"] = context.get("ticker_bid", context.get("bid", ""))
            row["ticker_ask"] = context.get("ticker_ask", context.get("ask", ""))
            row["ticker_mid"] = context.get("ticker_mid", context.get("mid", ""))
            row["ticker_spread_bps"] = context.get(
                "ticker_spread_bps", context.get("spread_bps", "")
            )
            for key in (
                "expected_spread_bps",
                "expected_spread_source",
                "expected_half_spread_bps",
                "entry_spread_bps",
                "entry_spread_source",
                "entry_vs_expected_spread_bps",
                "actual_exit_spread_bps",
                "actual_exit_ticker_spread_bps",
                "actual_exit_orderbook_spread_bps",
                "actual_exit_spread_source",
                "exit_vs_expected_spread_bps",
                "actual_exit_bid",
                "actual_exit_ask",
                "actual_exit_last",
                "close_execution_method",
                "close_execution_detail",
                "close_price_source",
                "close_trigger_type",
                "close_trigger_reference",
                "close_touch_side",
            ):
                row[key] = context.get(key, "")
            for key in (
                "sentinel_executable_price",
                "sentinel_executable_price_source",
                "sentinel_stop_distance_bps",
                "sentinel_stop_breach_overshoot_bps",
                "sentinel_pretrigger_enabled",
                "sentinel_pretrigger_buffer_bps",
                "sentinel_pretriggered",
                "last_lightweight_stop_sentinel_ts",
            ):
                row[key] = context.get(key, "")
            row["expected_fill_price"] = context.get("expected_fill_price", "")
            row["expected_fill_slippage_bps"] = context.get(
                "expected_fill_slippage_bps", ""
            )
            row["orderbook_slippage_bps"] = context.get(
                "orderbook_slippage_bps",
                context.get("expected_fill_slippage_bps", ""),
            )
            row["slippage_bps"] = context.get(
                "slippage_bps",
                context.get("expected_fill_slippage_bps", ""),
            )
            row["entry_gap_bps"] = context.get(
                "entry_gap_bps",
                context.get("adverse_signal_gap_bps", ""),
            )
            row["entry_slippage_proxy_bps"] = context.get(
                "entry_slippage_proxy_bps",
                context.get("expected_fill_slippage_bps", ""),
            )
            row["adverse_signal_gap_bps"] = context.get(
                "adverse_signal_gap_bps", ""
            )
            row["expected_total_entry_friction_bps"] = context.get(
                "expected_total_entry_friction_bps", ""
            )
            row["expected_friction_drag_bps"] = context.get(
                "expected_friction_drag_bps",
                context.get("expected_total_entry_friction_bps", ""),
            )
            for key in (
                "ev_haircut_bps",
                "ev_haircut_raw_live_entry_friction_bps",
                "ev_haircut_observed_spread_bps",
                "ev_haircut_observed_half_spread_bps",
                "ev_haircut_spread_baseline_bps",
                "ev_haircut_spread_baseline_source",
                "ev_haircut_half_spread_baseline_bps",
                "ev_haircut_spread_excess_bps",
                "ev_haircut_orderbook_slippage_bps",
                "ev_haircut_adverse_signal_gap_bps",
                "ev_haircut_observed_delay_slippage_bps",
                "ev_haircut_delay_slippage_baseline_bps",
                "ev_haircut_delay_slippage_excess_bps",
                "ev_haircut_expected_stop_exit_friction_bps",
                "ev_haircut_stop_exit_baseline_bps",
                "ev_haircut_stop_exit_excess_bps",
                "ev_haircut_stop_exit_source",
                "ev_haircut_contract",
                "ev_adjusted_entry_friction_bps",
                "ev_adjusted_net_return_before_friction",
                "ev_adjusted_net_return_after_friction",
                "ev_adjusted_calibrated_score",
                "ev_adjusted_rank_score",
                "ev_adjusted_source",
            ):
                row[key] = context.get(key, "")
            row["entry_delay_effect_bps"] = context.get("entry_delay_effect_bps", "")
            row["entry_delay_adverse_bps"] = context.get("entry_delay_adverse_bps", "")
            row["entry_delay_abs_bps"] = context.get("entry_delay_abs_bps", "")
            row["decision_to_entry_seconds"] = context.get(
                "decision_to_entry_seconds", ""
            )
            row["signal_close_to_entry_seconds"] = context.get(
                "signal_close_to_entry_seconds", ""
            )
            row["signal_to_entry_seconds"] = context.get("signal_to_entry_seconds", "")
            row["gross_to_net_friction_drag_bps"] = context.get(
                "gross_to_net_friction_drag_bps", ""
            )
            row["orderbook_side"] = context.get("orderbook_side", "")
            row["best_touch"] = context.get("best_touch", "")
            row["max_walk_price"] = context.get("max_walk_price", "")
            row["orderbook_capacity_quote_within_slippage"] = context.get(
                "orderbook_capacity_quote_within_slippage", ""
            )
            row["intended_quote_size"] = context.get("intended_quote_size", "")
            row["spread_weight"] = context.get("spread_weight", "")
            row["depth_weight"] = context.get("depth_weight", "")
            row["liquidity_capacity_weight"] = context.get(
                "liquidity_capacity_weight", ""
            )
            row["price_gap_penalty"] = context.get("price_gap_penalty", "")
            row["adjusted_rank_score"] = context.get("adjusted_rank_score", "")
            row["final_threshold"] = context.get("final_threshold", "")
            row["position_size_before_liquidity"] = context.get(
                "position_size_before_liquidity", ""
            )
            row["position_size_after_liquidity"] = context.get(
                "position_size_after_liquidity", ""
            )
            row["max_chase_bps"] = context.get("max_chase_bps", "")
            row["entry_limit_price"] = context.get("entry_limit_price", "")
            row["spread_proxy_pct"] = context.get("spread_proxy_pct", "")
            for col in TRADE_LOG_COLUMNS:
                if col in context and col not in row:
                    row[col] = context[col]

        row = self._persist_record(row)

        lifecycle_event = row.get("lifecycle_event") or action
        status = row.get("status") or ""
        tprint(
            f"Logged trade event: {lifecycle_event} status={status} "
            f"action={action} {side} {symbol} {size}@{price}"
        )

    def log_entry(
        self,
        symbol: str,
        side: str,
        size: float,
        price: Optional[float] = None,
        predictions: Optional[Dict[str, Any]] = None,
        features: Optional[Dict[str, Any]] = None,
        mode: str = "shadow",
        **extra: Any,
    ) -> None:
        """Compatibility wrapper used by inference runtime.

        Writes a lightweight row using legacy semantics while preserving
        parity-specific fields (strategy_id, calibrated_score, thresholds).
        """
        context = dict(predictions or {})
        context.update(features or {})
        context.update(extra or {})
        status = str(
            extra.get("status") or ("completed" if mode == "shadow" else "pending")
        )
        error = extra.get("error")
        self.log_trade_legacy(
            symbol=symbol,
            side=side,
            action="enter",
            size=float(size),
            price=price,
            context=context,
            mode=mode,
            status=status,
            error=str(error) if error else None,
        )


def log_trade_decision(
    logger: TradeLogger,
    symbol: str,
    side: str,
    action: str,
    size: float,
    price: Optional[float],
    predictions: Optional[Dict[str, float]] = None,
    features: Optional[Dict[str, float]] = None,
    mode: str = "shadow",
) -> None:
    """Log a trade decision.

    Convenience function.

    Args:
        logger: TradeLogger instance
        symbol: Trading symbol
        side: "long" or "short"
        action: "enter" or "exit"
        size: Position size
        price: Price
        predictions: Model predictions
        features: Feature values
        mode: Execution mode
    """
    # Build decision dict for legacy logging
    decision = {
        "symbol": symbol,
        "side": side,
        "action": action,
        "status": "completed" if mode == "shadow" else "pending",
    }

    # Build model results
    model_results = {}
    if predictions:
        model_results = {
            "alpha_preds": {
                "long_mr": predictions.get("alpha_mr"),
                "long_tf": predictions.get("alpha_tf"),
                "short_mr": predictions.get("alpha_mr"),
                "short_tf": predictions.get("alpha_tf"),
            },
            "meta_pred": predictions.get("meta_mr"),
            "meta_confidence": predictions.get("meta_confidence"),
            "position_size": predictions.get("position_size"),
            "disagreement_features": {
                "disagree_mr_std": predictions.get("disagreement_mr"),
                "disagree_tf_std": predictions.get("disagreement_tf"),
            },
        }

    # Build market data
    market_data = features or {}
    market_data["close"] = price

    # Config
    config = {"mode": mode, "run_id": logger.run_id}

    logger.log_trade(decision, model_results, market_data, config)
