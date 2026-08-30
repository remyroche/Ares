"""Live replay diagnostics for OOS-vs-live performance gaps."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

LIVE_REPLAY_COLUMNS = [
    "timestamp",
    "decision_ts",
    "signal_bar_ts",
    "symbol",
    "side",
    "strategy_id",
    "position_id",
    "trade_id",
    "lifecycle_entry_ts",
    "lifecycle_exit_ts",
    "was_traded",
    "portfolio_decision",
    "portfolio_reject_reason",
    "liquidity_reject_reason",
    "rank_score",
    "rank_percentile",
    "threshold",
    "signal_price",
    "decision_mid",
    "expected_fill_price",
    "realized_entry_price",
    "realized_exit_price",
    "entry_drag_bps",
    "exit_drag_bps",
    "fees_bps",
    "spread_bps",
    "expected_total_entry_friction_bps",
    "exit_reason",
    "holding_bars",
    "oos_expected_net_for_same_policy",
    "realized_net",
    "oos_expected_net_bps",
    "signal_forward_net_bps",
    "decision_mid_forward_net_bps",
    "fill_forward_net_bps",
    "realized_trade_net_bps",
    "primary_horizon_bars",
    "bar_minutes",
    "diagnostic_complete",
    "is_unresolved_trade",
    "unit_warning",
]

DECOMPOSITION_COLUMNS = [
    "live_mark_to_market_signal_outcome",
    "gap_oos_vs_realized_bps",
    "signal_outcome_gap_bps",
    "prediction_or_regime_gap_bps",
    "decision_delay_gap_bps",
    "entry_slippage_bps",
    "entry_execution_gap_bps",
    "extra_fees_funding_borrow_bps",
    "extra_cost_gap_bps",
    "stop_execution_mismatch_bps",
    "exit_policy_execution_gap_bps",
    "candidate_selection_mismatch_bps",
    "selection_gap_bps",
    "gap_explained_bps",
    "residual_model_drift_bps",
    "residual_bps",
]

_JOIN_KEYS = ["signal_bar_ts", "symbol", "side", "strategy_id"]


def _normalise_symbol(symbol: object) -> str:
    return str(symbol or "").upper().strip().replace(":USDT", "")


def _compact_symbol(symbol: object) -> str:
    return _normalise_symbol(symbol).replace("/", "").replace("_", "").replace("-", "")


def _normalise_join_id(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        num = float(text)
        if np.isfinite(num) and num.is_integer():
            return str(int(num))
    except (TypeError, ValueError):
        pass
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def _normalise_strategy_id(strategy_id: object) -> str:
    text = str(strategy_id or "").strip()
    for prefix in ("long_", "short_"):
        if text.startswith(prefix):
            return text[len(prefix) :]
    return text


def _first_present(row: pd.Series, names: Sequence[str], default=np.nan):
    for name in names:
        if name in row and pd.notna(row[name]) and row[name] != "":
            return row[name]
    return default


def _numeric(value, default=np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _side_sign(side: object) -> int:
    text = str(side or "").lower()
    return -1 if text in {"short", "sell", "-1"} else 1


def _side_return(start: float, end: float, side: object) -> float:
    start = _numeric(start)
    end = _numeric(end)
    if not np.isfinite(start) or not np.isfinite(end) or start <= 0 or end <= 0:
        return np.nan
    return (end / start - 1.0) if _side_sign(side) > 0 else (start / end - 1.0)


def _entry_adverse_slippage_bps(expected_fill, realized_entry, side) -> float:
    """Positive means a worse entry than expected."""
    expected_fill = _numeric(expected_fill)
    realized_entry = _numeric(realized_entry)
    if not np.isfinite(expected_fill) or not np.isfinite(realized_entry) or expected_fill <= 0:
        return np.nan
    if _side_sign(side) > 0:
        return (realized_entry / expected_fill - 1.0) * 10000.0
    return (expected_fill / realized_entry - 1.0) * 10000.0


def _exit_adverse_slippage_bps(expected_exit, realized_exit, side) -> float:
    """Positive means a worse exit than expected."""
    expected_exit = _numeric(expected_exit)
    realized_exit = _numeric(realized_exit)
    if not np.isfinite(expected_exit) or not np.isfinite(realized_exit) or expected_exit <= 0:
        return np.nan
    if _side_sign(side) > 0:
        return (expected_exit / realized_exit - 1.0) * 10000.0
    return (realized_exit / expected_exit - 1.0) * 10000.0


def _as_fraction(value: float) -> float:
    value = _numeric(value)
    if not np.isfinite(value):
        return np.nan
    return value / 100.0 if abs(value) > 2.0 else value


def _fraction_to_bps(value: float) -> float:
    value = _numeric(value)
    return value * 10000.0 if np.isfinite(value) else np.nan


def _bps_value(value: float) -> float:
    value = _numeric(value)
    return value if np.isfinite(value) else np.nan


def _unit_warnings(row: pd.Series) -> str:
    warnings = []
    for col in ("net_pnl_pct", "gross_pnl_pct", "gross_to_net_cost_pct", "expected_net"):
        if col not in row or pd.isna(row[col]) or row[col] == "":
            continue
        value = _numeric(row[col])
        if np.isfinite(value) and abs(value) > 1.0:
            warnings.append(f"{col}_abs_gt_1_check_units")
    return ";".join(warnings)


def _numeric_series(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(default, index=df.index, dtype=float)


def _truthy(value) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "filled", "traded", "accepted"}
    return bool(value)


def _is_failed_trade_lifecycle(row: pd.Series) -> bool:
    fields = []
    for col in (
        "action",
        "lifecycle_event",
        "status",
        "portfolio_decision",
        "portfolio_reject_reason",
        "liquidity_reject_reason",
        "order_error_category",
        "error",
    ):
        if col in row and pd.notna(row[col]):
            fields.append(str(row[col]).lower())
    text = " ".join(fields)
    return any(
        token in text
        for token in (
            "fail",
            "failed",
            "reject",
            "rejected",
            "refus",
            "error",
            "blocked",
        )
    )


def _normalise_times(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "decision_ts" not in out.columns and "timestamp" in out.columns:
        out["decision_ts"] = out["timestamp"]
    if "timestamp" not in out.columns and "decision_ts" in out.columns:
        out["timestamp"] = out["decision_ts"]
    if "signal_bar_ts" not in out.columns:
        out["signal_bar_ts"] = out.get("timestamp", pd.NaT)
    for col in ("timestamp", "decision_ts", "signal_bar_ts", "lifecycle_entry_ts", "lifecycle_exit_ts"):
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], utc=True, errors="coerce")
    if "symbol" in out.columns:
        out["symbol"] = out["symbol"].map(_normalise_symbol)
    if "strategy_id" in out.columns:
        out["strategy_id"] = out["strategy_id"].map(_normalise_strategy_id)
    for col in (
        "order_id",
        "exchange_order_id",
        "stop_order_id",
        "take_profit_order_id",
        "oco_id",
    ):
        if col in out.columns:
            out[col] = out[col].map(_normalise_join_id)
    return out


def _copy_first(row: pd.Series, out: dict, target: str, names: Sequence[str]) -> None:
    out[target] = _first_present(row, names)


def collapse_trade_lifecycle(trade_log: pd.DataFrame) -> pd.DataFrame:
    """Collapse TradeLogger lifecycle rows into one row per position_id."""
    if trade_log is None or trade_log.empty:
        return pd.DataFrame()
    src = _normalise_times(trade_log)
    if "position_id" not in src.columns or src["position_id"].isna().all():
        src["position_id"] = (
            src.get("symbol", "").astype(str)
            + "|"
            + src.get("side", "").astype(str)
            + "|"
            + src.get("strategy_id", "").astype(str)
            + "|"
            + src.get("timestamp", pd.NaT).astype(str)
        )

    action_all = src.get("action", pd.Series("", index=src.index)).astype(str).str.lower()
    event_all = src.get("lifecycle_event", pd.Series("", index=src.index)).astype(str).str.lower()
    entry_mask = (action_all == "enter") | event_all.str.contains("entry", na=False)
    stop_to_entry_position: dict[str, object] = {}
    order_to_entry_position: dict[str, object] = {}
    for _, row in src.loc[entry_mask].iterrows():
        pos = row.get("position_id")
        if pd.isna(pos) or pos == "":
            continue
        for key in ("stop_order_id", "take_profit_order_id", "oco_id"):
            val = row.get(key)
            join_id = _normalise_join_id(val)
            if join_id:
                stop_to_entry_position[join_id] = pos
        val = row.get("exchange_order_id")
        join_id = _normalise_join_id(val)
        if join_id:
            order_to_entry_position[join_id] = pos

    def _linked_position_id(row: pd.Series):
        pos = row.get("position_id")
        action_raw = row.get("action")
        event_raw = row.get("lifecycle_event")
        action = str(action_raw if pd.notna(action_raw) else "").lower()
        event = str(event_raw if pd.notna(event_raw) else "").lower()
        if (action == "exit" or "exit" in event) and stop_to_entry_position:
            for key in ("exchange_order_id", "stop_order_id", "position_id"):
                val = row.get(key)
                join_id = _normalise_join_id(val)
                if join_id in stop_to_entry_position:
                    return stop_to_entry_position[join_id]
            text = str(pos or "")
            if ":" in text:
                suffix = text.rsplit(":", 1)[-1]
                join_id = _normalise_join_id(suffix)
                if join_id in stop_to_entry_position:
                    return stop_to_entry_position[join_id]
        return pos

    src["_lifecycle_position_id"] = src.apply(_linked_position_id, axis=1)

    rows = []
    for position_id, grp in src.sort_values("timestamp").groupby(
        "_lifecycle_position_id", dropna=False
    ):
        action = grp.get("action", pd.Series("", index=grp.index)).astype(str).str.lower()
        event = grp.get("lifecycle_event", pd.Series("", index=grp.index)).astype(str).str.lower()
        status = grp.get("status", pd.Series("", index=grp.index)).astype(str).str.lower()
        entry_grp = grp[(action == "enter") | event.str.contains("entry", na=False)]
        exit_grp = grp[(action == "exit") | event.str.contains("exit", na=False) | status.isin({"closed", "completed"})]
        entry = (entry_grp.iloc[0] if not entry_grp.empty else grp.iloc[0])
        exit_ = (exit_grp.iloc[-1] if not exit_grp.empty else pd.Series(dtype=object))
        base = entry.to_dict()
        base["position_id"] = position_id
        if "exchange_order_id" in base:
            base["order_id"] = _normalise_join_id(base.get("exchange_order_id"))
        base["lifecycle_entry_ts"] = _first_present(entry, ["timestamp", "decision_ts"])
        base["was_traded"] = not _is_failed_trade_lifecycle(entry)
        if not exit_.empty:
            base["lifecycle_exit_ts"] = _first_present(exit_, ["timestamp", "decision_ts"])
            for col in (
                "realized_exit_price",
                "actual_exit_price",
                "exit_price",
                "exit_reason",
                "exit_reason_detail",
                "gross_pnl_pct",
                "net_pnl_pct",
                "net_pnl",
                "fees_bps",
                "realized_fee_bps",
                "realized_funding_bps",
                "realized_borrow_bps",
                "fees_amount",
                "gross_to_net_cost_pct",
                "holding_bars",
                "duration",
            ):
                if col in exit_ and pd.notna(exit_[col]) and exit_[col] != "":
                    base[col] = exit_[col]
        else:
            base["lifecycle_exit_ts"] = pd.NaT
            base.setdefault("realized_exit_price", np.nan)
        rows.append(base)
    return _normalise_times(pd.DataFrame(rows))


def _normalise_oos_policy(oos_policy: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if oos_policy is None or oos_policy.empty:
        return None
    oos = _normalise_times(oos_policy)
    aliases = {
        "oos_expected_net_bps": ["oos_expected_net_bps", "expected_net_bps", "policy_expected_net_bps"],
        "oos_expected_net_for_same_policy": [
            "oos_expected_net_for_same_policy",
            "oos_expected_net",
            "expected_net",
            "net_ret_equity",
            "policy_expected_net",
        ],
        "oos_selected": ["oos_selected", "selected", "would_select"],
    }
    for out_col, names in aliases.items():
        if out_col not in oos.columns:
            for name in names:
                if name in oos.columns:
                    oos[out_col] = oos[name]
                    break
    if "oos_expected_net_bps" not in oos.columns and "oos_expected_net_for_same_policy" in oos.columns:
        oos["oos_expected_net_bps"] = pd.to_numeric(oos["oos_expected_net_for_same_policy"], errors="coerce") * 10000.0
    keep = [c for c in _JOIN_KEYS + ["timestamp", "oos_expected_net_bps", "oos_expected_net_for_same_policy", "oos_selected"] if c in oos.columns]
    return oos.loc[:, keep].drop_duplicates([c for c in _JOIN_KEYS if c in keep], keep="last")


def _join_oos(
    src: pd.DataFrame,
    oos_policy: Optional[pd.DataFrame],
    *,
    oos_join_tolerance: pd.Timedelta,
) -> pd.DataFrame:
    oos = _normalise_oos_policy(oos_policy)
    if oos is None:
        return src
    out = src.copy()
    exact_keys = [c for c in _JOIN_KEYS if c in out.columns and c in oos.columns]
    if exact_keys:
        out = out.merge(oos, on=exact_keys, how="left", suffixes=("", "_oos"))
    needs = out.get("oos_expected_net_bps", pd.Series(np.nan, index=out.index)).isna()
    if not needs.any() or "timestamp" not in oos.columns:
        return out
    # Asof fallback against OOS timestamp, grouped by symbol/side/strategy_id.
    group_cols = [c for c in ["symbol", "side", "strategy_id"] if c in out.columns and c in oos.columns]
    for idx, row in out.loc[needs].iterrows():
        cand = oos.copy()
        for col in group_cols:
            cand = cand[cand[col].astype(str) == str(row[col])]
        if cand.empty or "timestamp" not in cand.columns:
            continue
        deltas = (cand["timestamp"] - row["signal_bar_ts"]).abs()
        best_idx = deltas.idxmin()
        if pd.notna(deltas.loc[best_idx]) and deltas.loc[best_idx] <= oos_join_tolerance:
            for col in ("oos_expected_net_bps", "oos_expected_net_for_same_policy", "oos_selected"):
                if col in cand.columns:
                    out.at[idx, col] = cand.loc[best_idx, col]
    return out


def _realized_cost_bps(row: pd.Series, default_expected_fee_bps: float, *, allow_legacy_cost_features: bool) -> tuple[float, float]:
    fee = _numeric(_first_present(row, ["realized_fee_bps", "fees_bps"]))
    if not np.isfinite(fee):
        fee = _numeric(_first_present(row, ["gross_to_net_cost_pct"])) * 10000.0
    if not np.isfinite(fee):
        fee = default_expected_fee_bps
    funding = _numeric(_first_present(row, ["realized_funding_bps", "funding_bps"]))
    borrow = _numeric(_first_present(row, ["realized_borrow_bps", "borrow_bps"]))
    if allow_legacy_cost_features:
        if not np.isfinite(funding):
            funding = _numeric(_first_present(row, ["funding_cost"]))
        if not np.isfinite(borrow):
            borrow = _numeric(_first_present(row, ["borrow_cost"]))
    total = np.nansum([fee, funding, borrow])
    extra = np.nansum([fee - default_expected_fee_bps, funding, borrow])
    return float(total), float(extra)


def _base_replay_table(
    live_trades: pd.DataFrame,
    *,
    oos_policy: Optional[pd.DataFrame],
    default_expected_fee_bps: float,
    oos_join_tolerance: pd.Timedelta,
    allow_legacy_cost_features: bool,
) -> pd.DataFrame:
    src = _normalise_times(live_trades)
    src = _join_oos(src, oos_policy, oos_join_tolerance=oos_join_tolerance)
    def _get_arr(cols, default):
        s_combined = None
        for col in cols:
            if col in src.columns:
                s = src[col]
                s_clean = s.copy()
                if s_clean.dtype == object:
                    mask = (s_clean == "") | s_clean.isna()
                else:
                    mask = s_clean.isna()
                s_clean.loc[mask] = np.nan
                if s_combined is None:
                    s_combined = s_clean
                else:
                    s_combined = s_combined.combine_first(s_clean)

        if s_combined is not None:
            if isinstance(default, np.ndarray):
                return s_combined.fillna(pd.Series(default, index=src.index)).values
            else:
                return s_combined.fillna(default).values

        if isinstance(default, np.ndarray):
            return default
        return np.full(len(src), default)

    side_arr = _get_arr(["side"], "long")
    signal_price_arr = _get_arr(["signal_price", "ohlcv_entry_price", "entry_price"], np.nan).astype(float)
    decision_mid_arr = _get_arr(["decision_mid", "ticker_mid", "signal_price"], signal_price_arr).astype(float)
    expected_fill_arr = _get_arr(["expected_fill_price", "expected_entry_price", "entry_price"], decision_mid_arr).astype(float)
    realized_entry_arr = _get_arr(["realized_entry_price", "actual_entry_price", "entry_price"], expected_fill_arr).astype(float)
    realized_exit_arr = _get_arr(["realized_exit_price", "actual_exit_price", "exit_price"], np.nan).astype(float)
    expected_exit_arr = _get_arr(["expected_exit_price", "oos_exit_price", "signal_exit_price"], np.nan).astype(float)

    fee_arr = _get_arr(["realized_fee_bps", "fees_bps"], np.nan).astype(float)
    gross_arr = _get_arr(["gross_to_net_cost_pct"], np.nan).astype(float)
    for i in range(len(fee_arr)):
        if not np.isfinite(fee_arr[i]):
            fee_arr[i] = gross_arr[i] * 10000.0 if np.isfinite(gross_arr[i]) else default_expected_fee_bps

    funding_arr = _get_arr(["realized_funding_bps", "funding_bps"], np.nan).astype(float)
    borrow_arr = _get_arr(["realized_borrow_bps", "borrow_bps"], np.nan).astype(float)
    if allow_legacy_cost_features:
        funding_cost_arr = _get_arr(["funding_cost"], np.nan).astype(float)
        borrow_cost_arr = _get_arr(["borrow_cost"], np.nan).astype(float)
        for i in range(len(funding_arr)):
            if not np.isfinite(funding_arr[i]):
                funding_arr[i] = funding_cost_arr[i]
            if not np.isfinite(borrow_arr[i]):
                borrow_arr[i] = borrow_cost_arr[i]

    realized_cost_bps_arr = np.nansum([fee_arr, funding_arr, borrow_arr], axis=0)
    extra_cost_bps_arr = np.nansum([fee_arr - default_expected_fee_bps, funding_arr, borrow_arr], axis=0)

    realized_net_bps_arr = _get_arr(["realized_trade_net_bps"], np.nan).astype(float)
    net_pct_arr = _get_arr(["net_pnl_pct", "net_ret_equity", "net_pnl"], np.nan).astype(float)

    oos_bps_arr = _get_arr(["oos_expected_net_bps"], np.nan).astype(float)
    oos_expected_net_arr = _get_arr(["oos_expected_net_for_same_policy", "oos_expected_net", "expected_net"], np.nan).astype(float)

    rank_score_arr = _get_arr(["rank_score", "adjusted_rank_score", "calibrated_score", "normalized_rank_score", "meta_pred"], np.nan).astype(float)
    rank_pct_arr = _get_arr(["rank_percentile", "sizer_rank_percentile", "base_rank_pct", "meta_train_rank_pct"], np.nan).astype(float)
    threshold_arr = _get_arr(["threshold", "final_threshold", "effective_threshold", "rank_threshold", "deployment_rank_threshold"], np.nan).astype(float)

    was_traded_arr = _get_arr(["was_traded"], True)

    ts_arr = _get_arr(["timestamp"], pd.NaT)
    decision_ts_arr = _get_arr(["decision_ts"], ts_arr)
    signal_bar_ts_arr = _get_arr(["signal_bar_ts"], ts_arr)

    sym_arr = _get_arr(["symbol"], "")
    strat_arr = _get_arr(["strategy_id"], "")
    pos_arr = _get_arr(["position_id"], "")
    trade_id_arr = _get_arr(["trade_id"], "")
    entry_ts_arr = _get_arr(["lifecycle_entry_ts"], ts_arr)
    exit_ts_arr = _get_arr(["lifecycle_exit_ts"], pd.NaT)

    pd_arr = _get_arr(["portfolio_decision"], np.where(was_traded_arr, "traded", "rejected"))
    pr_arr = _get_arr(["portfolio_reject_reason"], "")
    lr_arr = _get_arr(["liquidity_reject_reason"], "")

    spread_arr = _get_arr(["spread_bps", "ticker_spread_bps"], np.nan).astype(float)
    fric_arr = _get_arr(["expected_total_entry_friction_bps", "expected_fill_slippage_bps"], np.nan).astype(float)
    exit_reason_arr = _get_arr(["exit_reason", "exit_reason_detail"], "")
    holding_arr = _get_arr(["holding_bars", "duration", "duration_bars"], np.nan).astype(float)
    oos_sel_arr = _get_arr(["oos_selected"], np.nan)

    rows = []

    src_dicts = src.to_dict(orient="records")

    for i in range(len(src)):
        side = side_arr[i]
        realized_entry = realized_entry_arr[i]
        realized_exit = realized_exit_arr[i]
        realized_net_bps = realized_net_bps_arr[i]

        if not np.isfinite(realized_net_bps):
            if np.isfinite(net_pct_arr[i]):
                realized_net_bps = net_pct_arr[i] * 10000.0

        realized_cost_bps = realized_cost_bps_arr[i]
        if not np.isfinite(realized_net_bps):
            raw = _side_return(realized_entry, realized_exit, side)
            realized_net_bps = raw * 10000.0 - realized_cost_bps if np.isfinite(raw) else np.nan

        oos_bps = oos_bps_arr[i]
        if not np.isfinite(oos_bps):
            if np.isfinite(oos_expected_net_arr[i]):
                oos_bps = oos_expected_net_arr[i] * 10000.0

        expected_fill = expected_fill_arr[i]
        expected_exit = expected_exit_arr[i]

        entry_drag = _entry_adverse_slippage_bps(expected_fill, realized_entry, side)
        exit_drag = _exit_adverse_slippage_bps(expected_exit, realized_exit, side) if np.isfinite(expected_exit) else np.nan

        was_traded = _truthy(was_traded_arr[i])

        rows.append(
            {
                "timestamp": ts_arr[i],
                "decision_ts": decision_ts_arr[i],
                "signal_bar_ts": signal_bar_ts_arr[i],
                "symbol": sym_arr[i],
                "side": side,
                "strategy_id": strat_arr[i],
                "position_id": pos_arr[i],
                "trade_id": trade_id_arr[i],
                "lifecycle_entry_ts": entry_ts_arr[i],
                "lifecycle_exit_ts": exit_ts_arr[i],
                "was_traded": was_traded,
                "portfolio_decision": pd_arr[i],
                "portfolio_reject_reason": pr_arr[i],
                "liquidity_reject_reason": lr_arr[i],
                "rank_score": rank_score_arr[i],
                "rank_percentile": rank_pct_arr[i],
                "threshold": threshold_arr[i],
                "signal_price": signal_price_arr[i],
                "decision_mid": decision_mid_arr[i],
                "expected_fill_price": expected_fill,
                "realized_entry_price": realized_entry,
                "realized_exit_price": realized_exit,
                "entry_drag_bps": entry_drag,
                "exit_drag_bps": exit_drag,
                "fees_bps": realized_cost_bps,
                "spread_bps": spread_arr[i],
                "expected_total_entry_friction_bps": fric_arr[i],
                "exit_reason": str(exit_reason_arr[i]),
                "holding_bars": holding_arr[i],
                "oos_expected_net_for_same_policy": oos_bps / 10000.0 if np.isfinite(oos_bps) else np.nan,
                "realized_net": realized_net_bps / 10000.0 if np.isfinite(realized_net_bps) else np.nan,
                "oos_expected_net_bps": oos_bps,
                "realized_trade_net_bps": realized_net_bps,
                "realized_extra_cost_bps": extra_cost_bps_arr[i],
                "oos_assumed_cost_bps": default_expected_fee_bps,
                "oos_selected": oos_sel_arr[i],
                "primary_horizon_bars": np.nan,
                "bar_minutes": np.nan,
                "unit_warning": _unit_warnings(src_dicts[i]),
            }
        )
    return pd.DataFrame(rows)


def _column_for_symbol(df: pd.DataFrame, symbol: object) -> Optional[str]:
    if not isinstance(df, pd.DataFrame):
        return None
    if symbol in df.columns:
        return str(symbol)
    normalized = {_normalise_symbol(col): col for col in df.columns}
    if _normalise_symbol(symbol) in normalized:
        return normalized[_normalise_symbol(symbol)]
    compact = {_compact_symbol(col): col for col in df.columns}
    return compact.get(_compact_symbol(symbol))


def _future_price(close: pd.DataFrame, ts: pd.Timestamp, symbol: str, horizon: int):
    if not isinstance(close, pd.DataFrame) or close.empty:
        return np.nan
    col = _column_for_symbol(close, symbol)
    if col is None:
        return np.nan
    frame = close.copy()
    frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce")
    frame = frame.loc[pd.notna(frame.index)].sort_index()
    pos = frame.index.searchsorted(pd.Timestamp(ts), side="left")
    target = pos + int(horizon)
    if target >= len(frame.index):
        return np.nan
    return _numeric(frame.iloc[target][col])


def _future_extreme_return(
    prices: Optional[pd.DataFrame],
    ts: pd.Timestamp,
    symbol: str,
    horizon: int,
    start_price: float,
    side: object,
    *,
    favorable: bool,
) -> float:
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        return np.nan
    col = _column_for_symbol(prices, symbol)
    if col is None:
        return np.nan
    frame = prices.copy()
    frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce")
    frame = frame.loc[pd.notna(frame.index)].sort_index()
    pos = frame.index.searchsorted(pd.Timestamp(ts), side="left")
    start = pos + 1
    stop = min(len(frame.index), pos + int(horizon) + 1)
    if start >= stop:
        return np.nan
    window = pd.to_numeric(frame.iloc[start:stop][col], errors="coerce")
    if window.dropna().empty:
        return np.nan
    is_long = _side_sign(side) > 0
    if favorable:
        px = window.max() if is_long else window.min()
    else:
        px = window.min() if is_long else window.max()
    return _side_return(start_price, px, side)


def attach_forward_outcomes(
    replay: pd.DataFrame,
    *,
    close: pd.DataFrame,
    high: Optional[pd.DataFrame] = None,
    low: Optional[pd.DataFrame] = None,
    horizons: Sequence[int] = (1, 4, 8),
    bar_minutes: int = 15,
    price_col_signal: str = "signal_price",
    price_col_decision: str = "decision_mid",
    price_col_fill: str = "realized_entry_price",
    primary_horizon: int = 4,
) -> pd.DataFrame:
    """Attach side-aware close-to-close outcomes plus optional high/low MFE/MAE.

    The primary prediction-gap anchors use future close only; ``high``/``low`` are
    reserved for path diagnostics and populate MFE/MAE columns when supplied.
    """
    if replay is None or replay.empty:
        return replay
    out = _normalise_times(replay)
    for h in horizons:
        for prefix in ("signal", "decision_mid", "fill"):
            out[f"{prefix}_forward_return_{h}bar"] = np.nan
        for prefix in ("signal", "fill"):
            out[f"{prefix}_forward_mfe_{h}bar"] = np.nan
            out[f"{prefix}_forward_mae_{h}bar"] = np.nan

    ts_arr = out.get("signal_bar_ts", out.get("timestamp")).values
    sym_arr = out.get("symbol").values
    side_arr = out.get("side").values

    price_signal_arr = out.get(price_col_signal).values if price_col_signal in out.columns else np.full(len(out), np.nan)
    price_decision_arr = out.get(price_col_decision).values if price_col_decision in out.columns else np.full(len(out), np.nan)
    price_fill_arr = out.get(price_col_fill).values if price_col_fill in out.columns else np.full(len(out), np.nan)

    idx_arr = out.index.to_numpy()

    for i, idx in enumerate(idx_arr):
        ts = ts_arr[i]
        symbol = sym_arr[i]
        side = side_arr[i]
        psig = price_signal_arr[i]
        pdec = price_decision_arr[i]
        pfil = price_fill_arr[i]

        for h in horizons:
            fut = _future_price(close, ts, symbol, int(h))
            out.at[idx, f"signal_forward_return_{h}bar"] = _side_return(psig, fut, side)
            out.at[idx, f"decision_mid_forward_return_{h}bar"] = _side_return(pdec, fut, side)
            out.at[idx, f"fill_forward_return_{h}bar"] = _side_return(pfil, fut, side)
            out.at[idx, f"signal_forward_mfe_{h}bar"] = _future_extreme_return(
                high if _side_sign(side) > 0 else low,
                ts,
                symbol,
                int(h),
                psig,
                side,
                favorable=True,
            )
            out.at[idx, f"signal_forward_mae_{h}bar"] = _future_extreme_return(
                low if _side_sign(side) > 0 else high,
                ts,
                symbol,
                int(h),
                psig,
                side,
                favorable=False,
            )
            out.at[idx, f"fill_forward_mfe_{h}bar"] = _future_extreme_return(
                high if _side_sign(side) > 0 else low,
                ts,
                symbol,
                int(h),
                pfil,
                side,
                favorable=True,
            )
            out.at[idx, f"fill_forward_mae_{h}bar"] = _future_extreme_return(
                low if _side_sign(side) > 0 else high,
                ts,
                symbol,
                int(h),
                pfil,
                side,
                favorable=False,
            )
    h = int(primary_horizon)
    oos_cost = _numeric_series(out, "oos_assumed_cost_bps", 0.0).fillna(0.0)
    fees = _numeric_series(out, "fees_bps", 0.0).fillna(0.0)
    out["signal_forward_net_bps"] = _numeric_series(out, f"signal_forward_return_{h}bar") * 10000.0 - oos_cost
    out["decision_mid_forward_net_bps"] = _numeric_series(out, f"decision_mid_forward_return_{h}bar") * 10000.0 - oos_cost
    out["fill_forward_net_bps"] = _numeric_series(out, f"fill_forward_return_{h}bar") * 10000.0 - fees
    out["primary_horizon_bars"] = h
    out["bar_minutes"] = int(bar_minutes)
    return _apply_decomposition(out)


def _apply_decomposition(replay: pd.DataFrame) -> pd.DataFrame:
    out = replay.copy()
    for col in (
        "oos_expected_net_bps",
        "signal_forward_net_bps",
        "decision_mid_forward_net_bps",
        "fill_forward_net_bps",
        "realized_trade_net_bps",
    ):
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["prediction_or_regime_gap_bps"] = out["oos_expected_net_bps"] - out["signal_forward_net_bps"]
    out["decision_delay_gap_bps"] = out["signal_forward_net_bps"] - out["decision_mid_forward_net_bps"]
    out["entry_execution_gap_bps"] = out["decision_mid_forward_net_bps"] - out["fill_forward_net_bps"]
    out["exit_policy_execution_gap_bps"] = out["fill_forward_net_bps"] - out["realized_trade_net_bps"]
    out["extra_cost_gap_bps"] = _numeric_series(out, "realized_extra_cost_bps") - _numeric_series(out, "oos_assumed_cost_bps", 0.0).fillna(0.0)
    traded_raw = out["was_traded"] if "was_traded" in out.columns else pd.Series(True, index=out.index)
    traded = traded_raw.map(_truthy)
    out["selection_gap_bps"] = np.where(
        (~traded) & (out["signal_forward_net_bps"] > 0),
        out["signal_forward_net_bps"],
        np.nan,
    )
    total_gap = out["oos_expected_net_bps"] - out["realized_trade_net_bps"]
    components = out[
        [
            "prediction_or_regime_gap_bps",
            "decision_delay_gap_bps",
            "entry_execution_gap_bps",
            "exit_policy_execution_gap_bps",
        ]
    ].sum(axis=1, min_count=1)
    out["residual_bps"] = total_gap - components
    out["gap_oos_vs_realized_bps"] = total_gap
    out["signal_outcome_gap_bps"] = out["prediction_or_regime_gap_bps"]
    out["entry_slippage_bps"] = out.get("entry_drag_bps", np.nan)
    out["extra_fees_funding_borrow_bps"] = out.get("extra_cost_gap_bps", np.nan)
    exit_reason = out["exit_reason"] if "exit_reason" in out.columns else pd.Series("", index=out.index)
    exit_drag = out["exit_drag_bps"] if "exit_drag_bps" in out.columns else pd.Series(np.nan, index=out.index)
    out["stop_execution_mismatch_bps"] = np.where(
        exit_reason.astype(str).isin({"stop_loss", "trailing_stop", "early_invalidation"}),
        exit_drag,
        0.0,
    )
    out["candidate_selection_mismatch_bps"] = out["selection_gap_bps"]
    out["gap_explained_bps"] = components
    out["residual_model_drift_bps"] = out["residual_bps"]
    out["live_mark_to_market_signal_outcome"] = out["signal_forward_net_bps"] / 10000.0
    has_oos_expected = out["oos_expected_net_bps"].notna()
    has_signal_forward = out["signal_forward_net_bps"].notna()
    has_fill_forward = out["fill_forward_net_bps"].notna()
    has_realized_trade = out["realized_trade_net_bps"].notna()
    realized_exit = _numeric_series(out, "realized_exit_price")
    lifecycle_exit = pd.to_datetime(out.get("lifecycle_exit_ts", pd.Series(pd.NaT, index=out.index)), utc=True, errors="coerce")
    out["is_unresolved_trade"] = traded & has_signal_forward & realized_exit.isna() & lifecycle_exit.isna() & ~has_realized_trade
    out["diagnostic_complete"] = (
        has_oos_expected
        & has_signal_forward
        & ((~traded) | has_fill_forward)
        & ((~traded) | out["is_unresolved_trade"] | has_realized_trade)
    )
    return out


def build_live_replay_table(
    live_trades: pd.DataFrame,
    *,
    oos_policy: Optional[pd.DataFrame] = None,
    default_expected_fee_bps: float = 0.0,
    oos_join_tolerance: pd.Timedelta = pd.Timedelta("1min"),
    allow_legacy_cost_features: bool = False,
) -> pd.DataFrame:
    """Build one row per live trade and compute available decomposition fields."""
    if live_trades is None or live_trades.empty:
        return pd.DataFrame(columns=LIVE_REPLAY_COLUMNS + DECOMPOSITION_COLUMNS)
    src = collapse_trade_lifecycle(live_trades) if "lifecycle_event" in live_trades.columns or "action" in live_trades.columns else live_trades
    out = _base_replay_table(
        src,
        oos_policy=oos_policy,
        default_expected_fee_bps=default_expected_fee_bps,
        oos_join_tolerance=oos_join_tolerance,
        allow_legacy_cost_features=allow_legacy_cost_features,
    )
    out = _apply_decomposition(out)
    cols = list(dict.fromkeys(LIVE_REPLAY_COLUMNS + DECOMPOSITION_COLUMNS + list(out.columns)))
    return out.reindex(columns=cols)


def _coalesce_trade_suffix_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in list(out.columns):
        if not col.endswith("_trade"):
            continue
        base = col[:-6]
        if base in out.columns:
            if not out[col].notna().any():
                continue
            mask = out[base].isna() & out[col].notna()
            if bool(mask.any()):
                out.loc[mask, base] = out.loc[mask, col]
        else:
            out[base] = out[col]
    return out


def _merge_trade_lifecycle_on_nonempty_keys(
    candidates: pd.DataFrame,
    trades: pd.DataFrame,
    keys: Sequence[str],
) -> pd.DataFrame:
    out = candidates.copy()
    out["_trade_matched"] = False
    for key in keys:
        if key not in out.columns or key not in trades.columns:
            continue
        left_mask = (~out["_trade_matched"]) & out[key].map(_normalise_join_id).ne("")
        if not bool(left_mask.any()):
            continue
        right = trades[trades[key].map(_normalise_join_id).ne("")].drop_duplicates(
            key, keep="last"
        )
        if right.empty:
            continue
        merged = out.loc[left_mask].merge(
            right,
            on=key,
            how="left",
            suffixes=("", "_trade"),
        )
        merged.index = out.loc[left_mask].index
        matched = merged[[c for c in merged.columns if c.endswith("_trade")]].notna().any(axis=1)
        if not bool(matched.any()):
            continue
        for col in merged.columns:
            if col == key:
                continue
            if col not in out.columns:
                out[col] = pd.NA
            out.loc[merged.index[matched], col] = merged.loc[matched, col]
        out.loc[merged.index[matched], "_trade_matched"] = True
    out = out.drop(columns=["_trade_matched"])
    return out


def build_live_candidate_replay_table(
    prediction_ledger: pd.DataFrame,
    *,
    trade_log: Optional[pd.DataFrame] = None,
    oos_policy: Optional[pd.DataFrame] = None,
    forward_close: Optional[pd.DataFrame] = None,
    forward_high: Optional[pd.DataFrame] = None,
    forward_low: Optional[pd.DataFrame] = None,
    default_expected_fee_bps: float = 0.0,
) -> pd.DataFrame:
    """Build replay rows for traded and rejected live candidates."""
    if prediction_ledger is None or prediction_ledger.empty:
        return pd.DataFrame(columns=LIVE_REPLAY_COLUMNS + DECOMPOSITION_COLUMNS)
    candidates = _normalise_times(prediction_ledger)
    if "was_traded" not in candidates.columns:
        candidates["was_traded"] = candidates.get("portfolio_decision", "").astype(str).str.lower().isin({"trade", "traded", "accepted", "filled"})
    if trade_log is not None and not trade_log.empty:
        trades = collapse_trade_lifecycle(trade_log)
        for df in (candidates, trades):
            for col in ("order_id", "position_id", "trade_id"):
                if col in df.columns:
                    df[col] = df[col].map(_normalise_join_id)
        merge_keys = [
            c
            for c in ["position_id", "trade_id", "order_id"]
            if c in candidates.columns and c in trades.columns
        ]
        if merge_keys:
            candidates = _merge_trade_lifecycle_on_nonempty_keys(
                candidates,
                trades,
                merge_keys,
            )
        else:
            keys = [c for c in _JOIN_KEYS if c in candidates.columns and c in trades.columns]
            candidates = candidates.merge(trades, on=keys, how="left", suffixes=("", "_trade"))
        candidates = _coalesce_trade_suffix_columns(candidates)
    replay = _base_replay_table(
        candidates,
        oos_policy=oos_policy,
        default_expected_fee_bps=default_expected_fee_bps,
        oos_join_tolerance=pd.Timedelta("1min"),
        allow_legacy_cost_features=False,
    )
    replay = _apply_decomposition(replay)
    cols = list(dict.fromkeys(LIVE_REPLAY_COLUMNS + DECOMPOSITION_COLUMNS + list(replay.columns)))
    replay = replay.reindex(columns=cols)
    if forward_close is not None:
        replay = attach_forward_outcomes(replay, close=forward_close, high=forward_high, low=forward_low)
    return replay


def summarize_gap_decomposition(replay: pd.DataFrame) -> pd.DataFrame:
    """Return aggregate bps contribution summary for a replay table."""
    cols = [
        "gap_oos_vs_realized_bps",
        "prediction_or_regime_gap_bps",
        "decision_delay_gap_bps",
        "entry_execution_gap_bps",
        "exit_policy_execution_gap_bps",
        "extra_cost_gap_bps",
        "selection_gap_bps",
        "residual_bps",
    ]
    if replay is None or replay.empty:
        return pd.DataFrame(columns=["component", "mean_bps", "median_bps", "sum_bps", "non_null"])
    rows = []
    for col in cols:
        if col not in replay.columns:
            continue
        vals = pd.to_numeric(replay[col], errors="coerce")
        rows.append(
            {
                "component": col,
                "mean_bps": float(vals.mean()) if vals.notna().any() else np.nan,
                "median_bps": float(vals.median()) if vals.notna().any() else np.nan,
                "sum_bps": float(vals.sum()) if vals.notna().any() else np.nan,
                "non_null": int(vals.notna().sum()),
            }
        )
    return pd.DataFrame(rows)
