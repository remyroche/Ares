"""Reconcile policy execution assumptions against live inference observations.

This module intentionally separates two related checks:

1. Spread/slippage reconciliation: compare the simple-policy optimiser's
   expected friction proxy with what live inference observed around
   decision/order/fill time.
2. Decision replay reconciliation: run deployed portfolio-policy replay over
   live ledger candidates to detect rows that a backtest-style policy would
   accept while live inference did not open.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from extreme_price_movements.portfolio_policy_replay import (
    load_portfolio_policy_params,
    replay_candidates,
)


JOIN_KEYS = ["timestamp", "symbol", "side", "strategy_id"]


REPLAY_FIELD_GROUPS: tuple[dict[str, Any], ...] = (
    {
        "group": "decision_identity",
        "scope": "all",
        "required": (
            ("signal_bar_ts",),
            ("symbol",),
            ("side",),
            ("strategy_id",),
            ("portfolio_decision",),
        ),
    },
    {
        "group": "feature_prediction_replay",
        "scope": "all",
        "required": (
            ("base_model_features_json",),
            ("base_model_feature_values_json",),
            ("base_pred",),
            ("meta_pred",),
            ("calibrated_score", "raw_prediction_score"),
        ),
    },
    {
        "group": "rank_threshold_replay",
        "scope": "all",
        "required": (
            ("policy_rank_pct", "normalized_rank_score", "threshold_rank_score"),
            ("auction_rank_pct", "threshold_rank_score"),
            ("threshold_rank_score", "effective_threshold", "final_threshold", "initial_rank_threshold"),
            ("threshold_rank_score_source", "rank_score_source"),
            ("passed_rank_gate",),
        ),
    },
    {
        "group": "entry_timing_attribution",
        "scope": "traded",
        "required": (
            ("signal_bar_close_ts",),
            ("decision_ts",),
            ("theoretical_entry_price", "policy_entry_price", "signal_price"),
            ("expected_fill_price", "expected_entry_price"),
            ("realized_entry_price", "entry_price_actual"),
            ("signal_to_entry_seconds",),
            ("decision_to_entry_seconds",),
            ("hourly_close_to_latest_decision_price_bps", "signal_gap_bps"),
            ("decision_price_to_fill_bps", "actual_fill_vs_expected_bps"),
        ),
    },
    {
        "group": "spread_slippage_cost_attribution",
        "scope": "traded",
        "required": (
            ("ticker_spread_bps", "spread_bps", "spread_proxy_bps"),
            ("expected_fill_slippage_bps", "orderbook_live_slippage_bps", "orderbook_slippage_bps", "slippage_bps"),
            ("expected_total_entry_friction_bps", "max_entry_friction_bps"),
            ("fee_bps", "entry_fee_bps", "realized_fee_bps"),
            ("ev_haircut_bps", "expected_friction_drag_bps"),
        ),
    },
    {
        "group": "order_fill_identity",
        "scope": "traded",
        "required": (
            ("position_id",),
            ("order_id",),
            ("was_traded",),
            ("outcome_status", "portfolio_decision"),
        ),
    },
    {
        "group": "exact_portfolio_state_replay",
        "scope": "all",
        "required": (
            ("portfolio_state_snapshot_json", "open_positions_before_json", "active_positions_before_json"),
            ("portfolio_state_snapshot_hash", "portfolio_state_hash"),
            ("wallet_before", "wallet_value"),
            ("open_positions_before", "open_positions_before_count"),
            ("cooldowns_before_json", "recent_losing_trade_cooldown_state_json"),
            ("portfolio_priority",),
        ),
    },
)


def _read_table(path: str | Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    if p.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(p)
    raise ValueError(f"Unsupported table type: {p}")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(float(value)) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value) if not isinstance(value, (list, tuple, dict, str)) else False:
        return None
    return value


def _is_present(values: pd.Series) -> pd.Series:
    if values.empty:
        return pd.Series(dtype=bool)
    if pd.api.types.is_bool_dtype(values):
        return values.notna()
    if pd.api.types.is_numeric_dtype(values):
        return values.notna() & np.isfinite(pd.to_numeric(values, errors="coerce"))
    text = values.astype("string")
    return values.notna() & text.str.strip().ne("") & text.str.lower().ne("nan") & text.str.lower().ne("none")


def _alternative_present(df: pd.DataFrame, alternatives: Sequence[str]) -> pd.Series:
    present = pd.Series(False, index=df.index, dtype=bool)
    for col in alternatives:
        if col not in df.columns:
            continue
        present |= _is_present(df[col]).reindex(df.index, fill_value=False)
    return present


def _num(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def _first_numeric(df: pd.DataFrame, cols: Sequence[str], default: float = np.nan) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    for col in cols:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        out = out.where(out.notna(), vals)
    if not pd.isna(default):
        out = out.fillna(float(default))
    return out


def _normalise_side(value: Any, strategy_id: Any = "") -> str:
    raw = str(value or "").lower()
    if raw in {"1", "1.0", "long", "buy"}:
        return "long"
    if raw in {"-1", "-1.0", "short", "sell"}:
        return "short"
    sid = str(strategy_id or "").lower()
    if sid.startswith("short"):
        return "short"
    return "long"


def _normalise_ledger(ledger: pd.DataFrame) -> pd.DataFrame:
    out = ledger.copy()
    if "signal_bar_ts" in out.columns:
        out["timestamp"] = pd.to_datetime(out["signal_bar_ts"], utc=True, errors="coerce")
    else:
        out["timestamp"] = pd.to_datetime(out.get("timestamp"), utc=True, errors="coerce")
    out["symbol"] = out.get("symbol", pd.Series("", index=out.index)).astype(str)
    out["strategy_id"] = out.get("strategy_id", pd.Series("", index=out.index)).astype(str)
    side = out.get("side", pd.Series("", index=out.index))
    out["side"] = [
        _normalise_side(side_val, sid)
        for side_val, sid in zip(side, out["strategy_id"])
    ]
    out["_ledger_row_id"] = np.arange(len(out), dtype=np.int64)
    out["_join_seq"] = out.groupby(JOIN_KEYS, dropna=False).cumcount()
    return out


def _dedupe_latest_decision_rows(ledger: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty:
        return ledger
    order_ts = pd.to_datetime(
        ledger.get("decision_ts", ledger.get("timestamp")),
        utc=True,
        errors="coerce",
    )
    work = ledger.copy()
    work["_decision_order_ts"] = order_ts
    work = work.sort_values("_decision_order_ts", na_position="first")
    work = work.drop_duplicates(JOIN_KEYS, keep="last")
    work = work.drop(columns=["_decision_order_ts"])
    work["_join_seq"] = work.groupby(JOIN_KEYS, dropna=False).cumcount()
    return work


def _candidate_join_frame(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates is None or candidates.empty:
        return pd.DataFrame(columns=JOIN_KEYS)
    out = candidates.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["symbol"] = out["symbol"].astype(str)
    out["strategy_id"] = out["strategy_id"].astype(str)
    out["side"] = [
        _normalise_side(side, sid)
        for side, sid in zip(out.get("side", ""), out["strategy_id"])
    ]
    out["_join_seq"] = out.groupby(JOIN_KEYS, dropna=False).cumcount()
    join_cols = JOIN_KEYS + ["_join_seq"]
    cols = join_cols + [
        col
        for col in [
            "slippage_bps",
            "orderbook_slippage_bps",
            "expected_friction_bps",
            "entry_slippage_proxy_bps",
            "fees_bps",
            "entry_gap_bps",
            "price_gap_bps",
        ]
        if col in out.columns
    ]
    return out[cols].drop_duplicates(join_cols, keep="last")


def build_spread_slippage_reconciliation(
    prediction_ledger: pd.DataFrame,
    *,
    candidates: Optional[pd.DataFrame] = None,
    dedupe_latest: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ledger = _normalise_ledger(prediction_ledger)
    if dedupe_latest:
        ledger = _dedupe_latest_decision_rows(ledger)
    joined = ledger
    candidate_frame = _candidate_join_frame(
        candidates if candidates is not None else pd.DataFrame()
    )
    if not candidate_frame.empty:
        joined = ledger.merge(
            candidate_frame,
            on=JOIN_KEYS + ["_join_seq"],
            how="left",
            suffixes=("", "_policy_candidate"),
        )

    side_sign = np.where(joined["side"].astype(str).eq("short"), -1.0, 1.0)
    expected_policy_slippage = _first_numeric(
        joined,
        [
            "slippage_bps_policy_candidate",
            "entry_slippage_proxy_bps_policy_candidate",
            "entry_slippage_proxy_bps",
        ],
    )
    expected_policy_orderbook_slippage = _first_numeric(
        joined,
        [
            "orderbook_slippage_bps_policy_candidate",
            "slippage_bps_policy_candidate",
            "entry_slippage_proxy_bps_policy_candidate",
            "entry_slippage_proxy_bps",
        ],
    )
    expected_policy_friction = _first_numeric(
        joined,
        [
            "expected_friction_bps_policy_candidate",
            "expected_friction_bps",
            "expected_total_entry_friction_bps",
        ],
    )
    live_spread = _first_numeric(joined, ["ticker_spread_bps", "spread_bps", "spread_proxy_bps"])
    live_expected_slippage = _first_numeric(
        joined,
        [
            "expected_fill_slippage_bps",
            "orderbook_live_slippage_bps",
            "orderbook_slippage_bps",
            "slippage_bps",
        ],
    )
    live_total_friction = _first_numeric(
        joined,
        ["expected_total_entry_friction_bps", "max_entry_friction_bps"],
    )
    live_formula_friction = live_expected_slippage + live_spread.clip(lower=0.0) / 2.0
    live_total_friction = live_total_friction.where(live_total_friction.notna(), live_formula_friction)

    realized_entry = _num(joined, "realized_entry_price")
    expected_fill = _first_numeric(joined, ["expected_fill_price", "expected_entry_price"])
    theoretical_entry = _first_numeric(
        joined,
        ["theoretical_entry_price", "policy_entry_price", "signal_price"],
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        actual_fill_vs_expected_bps = side_sign * (
            realized_entry.to_numpy(dtype=float)
            / np.maximum(expected_fill.to_numpy(dtype=float), 1e-12)
            - 1.0
        ) * 10000.0
        actual_fill_vs_theoretical_bps = side_sign * (
            realized_entry.to_numpy(dtype=float)
            / np.maximum(theoretical_entry.to_numpy(dtype=float), 1e-12)
            - 1.0
        ) * 10000.0

    out = pd.DataFrame(
        {
            "timestamp": joined["timestamp"],
            "symbol": joined["symbol"],
            "side": joined["side"],
            "strategy_id": joined["strategy_id"],
            "portfolio_decision": joined.get("portfolio_decision"),
            "was_traded": joined.get("was_traded"),
            "expected_policy_slippage_bps": expected_policy_slippage,
            "expected_policy_orderbook_slippage_bps": expected_policy_orderbook_slippage,
            "expected_policy_friction_bps": expected_policy_friction,
            "live_spread_bps": live_spread,
            "live_expected_slippage_bps": live_expected_slippage,
            "live_total_entry_friction_bps": live_total_friction,
            "actual_fill_vs_expected_bps": actual_fill_vs_expected_bps,
            "actual_fill_vs_theoretical_bps": actual_fill_vs_theoretical_bps,
            "policy_vs_live_slippage_delta_bps": live_expected_slippage
            - expected_policy_slippage,
            "policy_vs_live_friction_delta_bps": live_total_friction
            - expected_policy_friction,
            "signal_to_entry_seconds": _num(joined, "signal_to_entry_seconds"),
            "decision_to_entry_seconds": _num(joined, "decision_to_entry_seconds"),
        }
    )
    summary = _summarise_spread_slippage(out)
    return out, summary


def _numeric_summary(values: pd.Series) -> dict[str, Any]:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return {"n": 0}
    return {
        "n": int(len(vals)),
        "mean": float(vals.mean()),
        "median": float(vals.median()),
        "p90": float(vals.quantile(0.90)),
        "max": float(vals.max()),
    }


def _summarise_spread_slippage(rows: pd.DataFrame) -> dict[str, Any]:
    cols = [
        "expected_policy_slippage_bps",
        "expected_policy_friction_bps",
        "live_spread_bps",
        "live_expected_slippage_bps",
        "live_total_entry_friction_bps",
        "actual_fill_vs_expected_bps",
        "actual_fill_vs_theoretical_bps",
        "policy_vs_live_slippage_delta_bps",
        "policy_vs_live_friction_delta_bps",
        "signal_to_entry_seconds",
    ]
    summary: dict[str, Any] = {
        "rows": int(len(rows)),
        "traded_rows": int(pd.Series(rows.get("was_traded", False)).fillna(False).astype(bool).sum()),
        "columns": {col: _numeric_summary(rows[col]) for col in cols if col in rows.columns},
    }
    if "strategy_id" in rows.columns:
        by_strategy = {}
        for strategy_id, group in rows.groupby("strategy_id", dropna=False):
            by_strategy[str(strategy_id)] = {
                "rows": int(len(group)),
                "traded_rows": int(pd.Series(group.get("was_traded", False)).fillna(False).astype(bool).sum()),
                "policy_vs_live_friction_delta_bps": _numeric_summary(
                    group.get("policy_vs_live_friction_delta_bps", pd.Series(dtype=float))
                ),
                "live_total_entry_friction_bps": _numeric_summary(
                    group.get("live_total_entry_friction_bps", pd.Series(dtype=float))
                ),
            }
        summary["by_strategy"] = by_strategy
    return summary


def build_ledger_replay_field_coverage(
    prediction_ledger: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ledger = _normalise_ledger(prediction_ledger)
    live_traded = ledger.apply(_live_traded, axis=1) if not ledger.empty else pd.Series(dtype=bool)
    rows: list[dict[str, Any]] = []
    for group in REPLAY_FIELD_GROUPS:
        group_name = str(group["group"])
        scope = str(group.get("scope", "all"))
        scoped = ledger
        if scope == "traded":
            scoped = ledger.loc[live_traded.reindex(ledger.index, fill_value=False)].copy()
        for alternatives in group["required"]:
            present = _alternative_present(scoped, list(alternatives))
            missing = ~present
            rows.append(
                {
                    "field_group": group_name,
                    "scope": scope,
                    "accepted_alternatives": "|".join(alternatives),
                    "rows_checked": int(len(scoped)),
                    "present_rows": int(present.sum()) if len(scoped) else 0,
                    "missing_rows": int(missing.sum()) if len(scoped) else 0,
                    "coverage_rate": float(present.mean()) if len(scoped) else np.nan,
                    "missing_symbols_sample": ",".join(
                        scoped.loc[missing, "symbol"].astype(str).drop_duplicates().head(12).tolist()
                    )
                    if len(scoped) and "symbol" in scoped.columns
                    else "",
                    "missing_decision_sample": ",".join(
                        scoped.loc[missing, "portfolio_decision"].astype(str).drop_duplicates().head(8).tolist()
                    )
                    if len(scoped) and "portfolio_decision" in scoped.columns
                    else "",
                }
            )
    report = pd.DataFrame(rows)
    if report.empty:
        summary = {
            "ledger_rows": int(len(ledger)),
            "live_traded_rows": int(live_traded.sum()) if len(live_traded) else 0,
            "field_checks": 0,
            "failed_field_checks": 0,
            "critical_missing_rows": 0,
        }
        return report, summary
    failed = report[pd.to_numeric(report["missing_rows"], errors="coerce").fillna(0) > 0]
    traded_failed = failed[failed["scope"].eq("traded")]
    state_group = next(
        (
            group
            for group in REPLAY_FIELD_GROUPS
            if str(group.get("group")) == "exact_portfolio_state_replay"
        ),
        None,
    )
    state_complete = pd.Series(False, index=ledger.index, dtype=bool)
    if state_group is not None and not ledger.empty:
        state_complete = pd.Series(True, index=ledger.index, dtype=bool)
        for alternatives in state_group["required"]:
            state_complete &= _alternative_present(ledger, list(alternatives))
    summary = {
        "ledger_rows": int(len(ledger)),
        "live_traded_rows": int(live_traded.sum()) if len(live_traded) else 0,
        "field_checks": int(len(report)),
        "failed_field_checks": int(len(failed)),
        "failed_traded_field_checks": int(len(traded_failed)),
        "critical_missing_rows": int(pd.to_numeric(failed["missing_rows"], errors="coerce").fillna(0).sum()),
        "exact_portfolio_state_replayable_rows": int(state_complete.sum()) if len(state_complete) else 0,
        "exact_portfolio_state_replayable_rate": (
            float(state_complete.mean()) if len(state_complete) else np.nan
        ),
        "exact_portfolio_state_replayable_traded_rows": (
            int((state_complete & live_traded.reindex(ledger.index, fill_value=False)).sum())
            if len(state_complete)
            else 0
        ),
        "worst_missing": failed.sort_values("missing_rows", ascending=False)
        .head(10)[["field_group", "scope", "accepted_alternatives", "missing_rows", "coverage_rate"]]
        .to_dict(orient="records"),
    }
    return report, summary


def _build_live_candidate_table(ledger: pd.DataFrame) -> pd.DataFrame:
    work = _normalise_ledger(ledger)
    rank = _first_numeric(
        work,
        ["threshold_rank_score", "auction_rank_pct", "normalized_rank_score", "policy_rank_pct"],
    )
    base_threshold = _first_numeric(
        work,
        ["final_threshold", "effective_threshold", "base_strategy_threshold", "initial_rank_threshold"],
        default=1.0,
    )
    entry = _first_numeric(
        work,
        [
            "theoretical_entry_price",
            "policy_entry_price",
            "expected_entry_price",
            "realized_entry_price",
        ],
        default=1.0,
    ).fillna(1.0)
    timestamp = work["timestamp"]
    holding_bars = pd.Series(4.0, index=work.index)
    gross_return = pd.Series(0.01, index=work.index)
    side_sign = np.where(work["side"].eq("short"), -1.0, 1.0)
    out = pd.DataFrame(
        {
            "_ledger_row_id": work["_ledger_row_id"],
            "_join_seq": work["_join_seq"],
            "timestamp": timestamp,
            "symbol": work["symbol"],
            "side": work["side"],
            "strategy_id": work["strategy_id"],
            "normalized_rank_score": rank,
            "strategy_rank_pct": _first_numeric(work, ["policy_rank_pct", "historical_rank_pct"]),
            "base_strategy_threshold": base_threshold,
            "calibrated_score": _first_numeric(
                work,
                ["raw_prediction_score", "meta_pred", "base_pred", "threshold_rank_score"],
                default=0.0,
            ).fillna(0.0),
            "entry_price": entry,
            "exit_timestamp": timestamp + pd.to_timedelta(holding_bars * 15, unit="m"),
            "exit_price": entry * (1.0 + side_sign * gross_return),
            "net_return": gross_return - 0.001,
            "gross_return": gross_return,
            "fees_bps": 10.0,
            "slippage_bps": _first_numeric(work, ["entry_slippage_proxy_bps"], default=0.0).fillna(0.0),
            "holding_bars": holding_bars,
            "simple_policy_exit_reason": "decision_replay_placeholder",
            "price_gap_bps": _first_numeric(
                work,
                ["entry_gap_bps", "adverse_signal_gap_bps", "price_gap_bps"],
                default=0.0,
            ).fillna(0.0).clip(lower=0.0),
            "expected_friction_bps": _first_numeric(
                work,
                ["expected_total_entry_friction_bps", "expected_friction_bps"],
                default=0.0,
            ).fillna(0.0),
            "liquidity_capacity_weight": 1.0,
            "market_mode": "perps",
        }
    )
    return out.dropna(subset=["timestamp", "symbol", "strategy_id", "normalized_rank_score"])


def _live_traded(row: pd.Series) -> bool:
    if "was_traded" in row and pd.notna(row["was_traded"]):
        try:
            return bool(row["was_traded"])
        except Exception:
            pass
    return str(row.get("portfolio_decision", "")).lower() in {"trade", "traded", "accepted"}


def _bool_series(values: pd.Series) -> pd.Series:
    def _one(value: Any) -> Any:
        if pd.isna(value):
            return np.nan
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        text = str(value).strip().lower()
        if text in {"true", "1", "1.0", "yes", "y"}:
            return True
        if text in {"false", "0", "0.0", "no", "n"}:
            return False
        return np.nan

    return values.map(_one)


def _explanation(row: pd.Series) -> str:
    if bool(row.get("live_traded", False)):
        return "live_traded"
    reason = str(
        row.get("portfolio_reject_reason")
        or row.get("liquidity_reject_reason")
        or row.get("portfolio_decision")
        or ""
    )
    if "rank_below" in reason or "below_dynamic_threshold" in reason:
        return "rank_threshold"
    if "stale" in reason:
        return "live_stale_signal_or_data_gate"
    if "spread" in reason or "slippage" in reason or "friction" in reason:
        return "live_spread_slippage_gate"
    if "min_notional" in reason or "position_size" in reason or "wallet" in reason:
        return "live_sizing_or_wallet_gate"
    if "missing_policy_rank" in reason or "rank_reference" in reason:
        return "rank_reference_unavailable"
    if reason:
        return f"live_reject:{reason}"
    return "unexplained_live_not_traded"


def _add_direct_gate_reconciliation(merged: pd.DataFrame) -> pd.DataFrame:
    out = merged.copy()
    rank = _first_numeric(
        out,
        ["threshold_rank_score", "auction_rank_pct", "normalized_rank_score", "policy_rank_pct"],
    )
    threshold = _first_numeric(out, ["final_threshold", "effective_threshold", "initial_rank_threshold"])
    direct = rank >= threshold
    if "passed_rank_gate" in out.columns:
        passed = _bool_series(out["passed_rank_gate"])
        direct = direct.where(passed.isna(), passed.astype("boolean"))
    out["direct_rank_gate_would_open"] = direct.fillna(False).astype(bool)
    out["direct_rank_gate_rank"] = rank
    out["direct_rank_gate_threshold"] = threshold
    out["direct_rank_gate_matches_live_trade"] = (
        out["direct_rank_gate_would_open"].astype(bool)
        == out["live_traded"].astype(bool)
    )
    out["direct_rank_gate_gap_explanation"] = out.apply(
        lambda row: "match"
        if bool(row["direct_rank_gate_matches_live_trade"])
        else _explanation(row),
        axis=1,
    )
    return out


def build_live_decision_replay_reconciliation(
    prediction_ledger: pd.DataFrame,
    *,
    portfolio_policy_config_path: str | Path,
    initial_wallet: float = 10_000.0,
    dedupe_latest: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ledger = _normalise_ledger(prediction_ledger)
    if dedupe_latest:
        ledger = _dedupe_latest_decision_rows(ledger)
    candidates = _build_live_candidate_table(ledger)
    params = load_portfolio_policy_params(portfolio_policy_config_path)
    params = replace(
        params,
        global_threshold_floor=0.0,
        occupancy_threshold_alpha=0.0,
        threshold_viability_margin=0.0,
    )
    decisions, _, _ = replay_candidates(
        candidates,
        params,
        mode="global_auction",
        initial_wallet=float(initial_wallet),
        market_mode="perps",
    )
    key = ["timestamp", "symbol", "side", "strategy_id"]
    decision_cols = key + [
        "normalized_rank_score",
        "base_threshold",
        "accepted",
        "rejection_reason",
        "dynamic_threshold",
        "portfolio_priority",
        "position_size",
        "open_positions_before",
        "open_positions_after",
    ]
    replay_join = decisions.copy()
    replay_join["timestamp"] = pd.to_datetime(
        replay_join["timestamp"], utc=True, errors="coerce"
    )
    candidate_map = candidates[
        key + ["_ledger_row_id", "normalized_rank_score", "base_strategy_threshold"]
    ].copy()
    candidate_map["timestamp"] = pd.to_datetime(
        candidate_map["timestamp"], utc=True, errors="coerce"
    )
    for frame in (replay_join, candidate_map):
        frame["symbol"] = frame["symbol"].astype(str)
        frame["side"] = frame["side"].astype(str)
        frame["strategy_id"] = frame["strategy_id"].astype(str)
    replay_join["_rank_key"] = pd.to_numeric(
        replay_join["normalized_rank_score"], errors="coerce"
    ).round(12)
    replay_join["_threshold_key"] = pd.to_numeric(
        replay_join["base_threshold"], errors="coerce"
    ).round(12)
    candidate_map["_rank_key"] = pd.to_numeric(
        candidate_map["normalized_rank_score"], errors="coerce"
    ).round(12)
    candidate_map["_threshold_key"] = pd.to_numeric(
        candidate_map["base_strategy_threshold"], errors="coerce"
    ).round(12)
    replay_join = replay_join.merge(
        candidate_map[key + ["_rank_key", "_threshold_key", "_ledger_row_id"]],
        on=key + ["_rank_key", "_threshold_key"],
        how="left",
    )
    merged = ledger.merge(
        replay_join[
            [
                "_ledger_row_id",
                "accepted",
                "rejection_reason",
                "dynamic_threshold",
                "portfolio_priority",
                "position_size",
                "open_positions_before",
                "open_positions_after",
            ]
        ],
        on="_ledger_row_id",
        how="left",
        suffixes=("", "_replay"),
    )
    merged["live_traded"] = merged.apply(_live_traded, axis=1)
    state_required = (
        ("portfolio_state_snapshot_json", "open_positions_before_json", "active_positions_before_json"),
        ("portfolio_state_snapshot_hash", "portfolio_state_hash"),
        ("wallet_before", "wallet_value"),
        ("open_positions_before", "open_positions_before_count"),
        ("cooldowns_before_json", "recent_losing_trade_cooldown_state_json"),
        ("portfolio_priority",),
    )
    replayable = pd.Series(True, index=merged.index, dtype=bool)
    for alternatives in state_required:
        replayable &= _alternative_present(merged, list(alternatives))
    merged["exact_portfolio_state_replayable"] = replayable
    merged["replay_accepted"] = merged["accepted"].fillna(False).astype(bool)
    merged["decision_match"] = merged["live_traded"] == merged["replay_accepted"]
    merged["replay_live_gap_class"] = merged.apply(
        lambda row: "match"
        if bool(row["decision_match"])
        else (
            "replay_accept_live_reject"
            if bool(row["replay_accepted"])
            else "live_accept_replay_reject"
        ),
        axis=1,
    )
    merged["replay_live_gap_explanation"] = merged.apply(_explanation, axis=1)
    merged = _add_direct_gate_reconciliation(merged)
    summary = {
        "rows": int(len(merged)),
        "candidate_rows": int(len(candidates)),
        "replay_rows": int(len(decisions)),
        "live_traded": int(merged["live_traded"].sum()),
        "replay_accepted": int(merged["replay_accepted"].sum()),
        "decision_matches": int(merged["decision_match"].sum()),
        "decision_mismatches": int((~merged["decision_match"]).sum()),
        "gap_classes": merged["replay_live_gap_class"].value_counts(dropna=False).to_dict(),
        "gap_explanations": merged["replay_live_gap_explanation"].value_counts(dropna=False).to_dict(),
        "replay_rejection_reasons": merged["rejection_reason"].value_counts(dropna=False).to_dict(),
        "live_portfolio_reasons": merged.get(
            "portfolio_reject_reason", pd.Series(dtype=object)
        ).value_counts(dropna=False).to_dict(),
        "direct_rank_gate_would_open": int(merged["direct_rank_gate_would_open"].sum()),
        "direct_rank_gate_matches": int(
            merged["direct_rank_gate_matches_live_trade"].sum()
        ),
        "direct_rank_gate_mismatches": int(
            (~merged["direct_rank_gate_matches_live_trade"]).sum()
        ),
        "direct_rank_gate_gap_explanations": merged[
            "direct_rank_gate_gap_explanation"
        ].value_counts(dropna=False).to_dict(),
        "exact_portfolio_state_replayable_rows": int(
            merged["exact_portfolio_state_replayable"].sum()
        ),
        "exact_portfolio_state_replayable_traded_rows": int(
            (
                merged["exact_portfolio_state_replayable"].astype(bool)
                & merged["live_traded"].astype(bool)
            ).sum()
        ),
        "exact_portfolio_state_replayable_note": (
            "Rows without persisted portfolio state are candidate/rank replay rows, "
            "not exact stateful portfolio replay proof."
        ),
    }
    return merged, summary


def _render_markdown(
    *,
    spread_summary: dict[str, Any],
    decision_summary: dict[str, Any],
    field_summary: dict[str, Any],
) -> str:
    return "\n".join(
        [
            "# Execution and Decision Reconciliation",
            "",
            "## Spread / Slippage",
            f"- Rows: `{spread_summary.get('rows', 0)}`",
            f"- Traded rows: `{spread_summary.get('traded_rows', 0)}`",
            f"- Policy vs live friction delta: `{spread_summary.get('columns', {}).get('policy_vs_live_friction_delta_bps', {})}`",
            f"- Live total entry friction: `{spread_summary.get('columns', {}).get('live_total_entry_friction_bps', {})}`",
            "",
            "## Backtest / Live Open Decision",
            f"- Ledger rows: `{decision_summary.get('rows', 0)}`",
            f"- Live traded: `{decision_summary.get('live_traded', 0)}`",
            f"- Replay accepted: `{decision_summary.get('replay_accepted', 0)}`",
            f"- Decision mismatches: `{decision_summary.get('decision_mismatches', 0)}`",
            f"- Gap classes: `{decision_summary.get('gap_classes', {})}`",
            f"- Gap explanations: `{decision_summary.get('gap_explanations', {})}`",
            f"- Direct rank-gate would open: `{decision_summary.get('direct_rank_gate_would_open', 0)}`",
            f"- Direct rank-gate mismatches: `{decision_summary.get('direct_rank_gate_mismatches', 0)}`",
            f"- Direct rank-gate gap explanations: `{decision_summary.get('direct_rank_gate_gap_explanations', {})}`",
            f"- Exact portfolio-state replayable rows: `{decision_summary.get('exact_portfolio_state_replayable_rows', 0)}`",
            f"- Exact portfolio-state replayable traded rows: `{decision_summary.get('exact_portfolio_state_replayable_traded_rows', 0)}`",
            "",
            "## Replay Field Coverage",
            f"- Ledger rows: `{field_summary.get('ledger_rows', 0)}`",
            f"- Live traded rows: `{field_summary.get('live_traded_rows', 0)}`",
            f"- Exact portfolio-state replayable rows: `{field_summary.get('exact_portfolio_state_replayable_rows', 0)}`",
            f"- Exact portfolio-state replayable rate: `{field_summary.get('exact_portfolio_state_replayable_rate', 0)}`",
            f"- Failed field checks: `{field_summary.get('failed_field_checks', 0)}`",
            f"- Failed traded-field checks: `{field_summary.get('failed_traded_field_checks', 0)}`",
            f"- Critical missing rows: `{field_summary.get('critical_missing_rows', 0)}`",
            f"- Worst missing: `{field_summary.get('worst_missing', [])}`",
            "",
            "Note: decision replay uses live ledger candidates and deployed portfolio-policy gates. "
            "It is a final gate parity audit, not a PnL backtest.",
            "",
        ]
    )


def run_reconciliation(
    *,
    prediction_ledger_path: str | Path,
    portfolio_policy_config_path: str | Path,
    output_dir: str | Path,
    candidate_path: str | Path | None = None,
    initial_wallet: float = 10_000.0,
) -> dict[str, Any]:
    ledger = _read_table(prediction_ledger_path)
    candidates = _read_table(candidate_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spread_rows, spread_summary = build_spread_slippage_reconciliation(
        ledger,
        candidates=candidates,
    )
    field_rows, field_summary = build_ledger_replay_field_coverage(ledger)
    decision_rows, decision_summary = build_live_decision_replay_reconciliation(
        ledger,
        portfolio_policy_config_path=portfolio_policy_config_path,
        initial_wallet=initial_wallet,
    )
    spread_rows.to_csv(out_dir / "spread_slippage_reconciliation.csv", index=False)
    field_rows.to_csv(out_dir / "ledger_replay_field_coverage.csv", index=False)
    decision_rows.to_csv(out_dir / "live_decision_replay_reconciliation.csv", index=False)
    (out_dir / "spread_slippage_reconciliation.json").write_text(
        json.dumps(_json_safe(spread_summary), indent=2),
        encoding="utf-8",
    )
    (out_dir / "ledger_replay_field_coverage.json").write_text(
        json.dumps(_json_safe(field_summary), indent=2),
        encoding="utf-8",
    )
    (out_dir / "live_decision_replay_reconciliation.json").write_text(
        json.dumps(_json_safe(decision_summary), indent=2),
        encoding="utf-8",
    )
    markdown = _render_markdown(
        spread_summary=spread_summary,
        decision_summary=decision_summary,
        field_summary=field_summary,
    )
    (out_dir / "execution_and_decision_reconciliation.md").write_text(
        markdown,
        encoding="utf-8",
    )
    return {
        "spread_slippage": spread_summary,
        "field_coverage": field_summary,
        "decision_replay": decision_summary,
        "output_dir": str(out_dir),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-ledger", required=True)
    parser.add_argument("--portfolio-policy-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate-path")
    parser.add_argument("--initial-wallet", type=float, default=10_000.0)
    args = parser.parse_args(argv)
    result = run_reconciliation(
        prediction_ledger_path=args.prediction_ledger,
        portfolio_policy_config_path=args.portfolio_policy_config,
        output_dir=args.output_dir,
        candidate_path=args.candidate_path,
        initial_wallet=args.initial_wallet,
    )
    print(json.dumps(_json_safe(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
