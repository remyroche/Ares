#!/usr/bin/env python3
"""Bridge live execution rows to policy replay/provenance diagnostics.

This is diagnostic-only. It does not place orders and does not mutate live
state. Its main purpose is to make execution-vs-policy gaps measurable from the
prediction ledger plus the trade logger.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_RUN_ID = "20260617_090000_no_mkt4_labelhpo_final_fit"
DEFAULT_DATA_ROOT = "data_perp"
DEFAULT_TRADE_LOG = "inference_trades.csv"


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    text = str(value).strip()
    return bool(text) and text.lower() not in {"nan", "none", "null"}


def _json_loads(value: Any, default: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if not _is_present(value):
        return default
    try:
        return json.loads(str(value))
    except Exception:
        return default


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    if value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    return pd.read_csv(path)


def _normalise_time_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in (
        "timestamp",
        "decision_ts",
        "signal_bar_ts",
        "signal_bar_close_ts",
        "entry_time",
        "exit_time",
        "lifecycle_entry_ts",
        "lifecycle_exit_ts",
    ):
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], utc=True, errors="coerce")
    return out


def _normalise_symbol(value: Any) -> str:
    return str(value or "").upper().strip()


def _normalise_side(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text if text in {"long", "short"} else text


def _first_present(row: pd.Series, names: tuple[str, ...], default: Any = np.nan) -> Any:
    for name in names:
        if name in row and _is_present(row.get(name)):
            return row.get(name)
    return default


def _choose(entry: pd.Series, exit_: pd.Series, names: tuple[str, ...], *, exit_first: bool = False) -> Any:
    rows = (exit_, entry) if exit_first else (entry, exit_)
    for row in rows:
        if row is None or row.empty:
            continue
        value = _first_present(row, names)
        if _is_present(value):
            return value
    return np.nan


def _action_or_event_mask(frame: pd.DataFrame, words: tuple[str, ...]) -> pd.Series:
    action = frame.get("action", pd.Series("", index=frame.index)).astype(str).str.lower()
    event = frame.get("lifecycle_event", pd.Series("", index=frame.index)).astype(str).str.lower()
    status = frame.get("status", pd.Series("", index=frame.index)).astype(str).str.lower()
    mask = pd.Series(False, index=frame.index)
    for word in words:
        mask |= action.eq(word) | event.str.contains(word, na=False) | status.eq(word)
    return mask


def collapse_trade_rows(trade_log: pd.DataFrame, *, run_id: str | None = None) -> pd.DataFrame:
    if trade_log.empty:
        return pd.DataFrame()
    work = _normalise_time_columns(trade_log)
    if run_id and "run_id" in work.columns:
        work = work[work["run_id"].astype(str).eq(str(run_id))].copy()
    if work.empty:
        return pd.DataFrame()
    if "position_id" not in work.columns:
        work["position_id"] = (
            work.get("symbol", "").astype(str)
            + "|"
            + work.get("side", "").astype(str)
            + "|"
            + work.get("strategy_id", "").astype(str)
            + "|"
            + work.get("timestamp", pd.NaT).astype(str)
        )
    entry_mask = _action_or_event_mask(work, ("enter", "entry"))
    exit_mask = _action_or_event_mask(work, ("exit", "closed", "close", "completed"))
    rows: list[dict[str, Any]] = []
    for position_id, group in work.sort_values("timestamp").groupby("position_id", dropna=False):
        entry_group = group[entry_mask.reindex(group.index, fill_value=False)]
        exit_group = group[exit_mask.reindex(group.index, fill_value=False)]
        entry = entry_group.iloc[0] if not entry_group.empty else group.iloc[0]
        exit_ = exit_group.iloc[-1] if not exit_group.empty else pd.Series(dtype=object)
        row = {
            "position_id": position_id,
            "trade_id": _choose(entry, exit_, ("trade_id",)),
            "run_id": _choose(entry, exit_, ("run_id",)),
            "symbol": _normalise_symbol(_choose(entry, exit_, ("symbol",))),
            "side": _normalise_side(_choose(entry, exit_, ("side",))),
            "strategy_id": str(_choose(entry, exit_, ("strategy_id",)) or ""),
            "decision_ts": _choose(entry, exit_, ("decision_ts",)),
            "signal_bar_ts": _choose(entry, exit_, ("signal_bar_ts",)),
            "entry_ts": _choose(entry, exit_, ("entry_time", "timestamp")),
            "exit_ts": _choose(entry, exit_, ("exit_time", "timestamp"), exit_first=True),
            "status": _choose(entry, exit_, ("status",), exit_first=True),
            "entry_price": _safe_float(_choose(entry, exit_, ("actual_entry_price", "realized_entry_price", "entry_price"))),
            "exit_price": _safe_float(_choose(entry, exit_, ("actual_exit_price", "realized_exit_price", "exit_price"), exit_first=True)),
            "expected_spread_bps": _safe_float(_choose(entry, exit_, ("expected_spread_bps",))),
            "entry_spread_bps": _safe_float(_choose(entry, exit_, ("entry_spread_bps",))),
            "actual_exit_spread_bps": _safe_float(_choose(entry, exit_, ("actual_exit_spread_bps",), exit_first=True)),
            "entry_vs_expected_spread_bps": _safe_float(
                _choose(entry, exit_, ("entry_vs_expected_spread_bps",))
            ),
            "exit_vs_expected_spread_bps": _safe_float(
                _choose(entry, exit_, ("exit_vs_expected_spread_bps",), exit_first=True)
            ),
            "requested_policy_stop": _safe_float(
                _choose(entry, exit_, ("requested_policy_stop", "policy_stop_price", "stop_price"), exit_first=True)
            ),
            "final_placed_stop": _safe_float(
                _choose(entry, exit_, ("final_placed_stop", "exchange_stop_price", "stop_price"), exit_first=True)
            ),
            "exit_vs_policy_stop_bps": _safe_float(
                _choose(entry, exit_, ("exit_vs_policy_stop_bps",), exit_first=True)
            ),
            "close_execution_method": _choose(entry, exit_, ("close_execution_method",), exit_first=True),
            "close_price_source": _choose(entry, exit_, ("close_price_source",), exit_first=True),
            "exit_reason": _choose(entry, exit_, ("exit_reason", "reason"), exit_first=True),
            "gross_pnl_pct": _safe_float(_choose(entry, exit_, ("gross_pnl_pct",), exit_first=True)),
            "net_pnl_pct": _safe_float(_choose(entry, exit_, ("net_pnl_pct", "net_pnl_pct_estimated"), exit_first=True)),
            "gross_pnl_amount": _safe_float(_choose(entry, exit_, ("gross_pnl_amount", "gross_pnl"), exit_first=True)),
            "net_pnl_amount": _safe_float(_choose(entry, exit_, ("net_pnl_amount", "net_pnl", "net_pnl_estimated"), exit_first=True)),
            "entry_notional_quote": _safe_float(_choose(entry, exit_, ("entry_notional_quote", "quote_size"))),
            "fees_verified": _choose(entry, exit_, ("fees_verified",), exit_first=True),
            "fees_estimated": _choose(entry, exit_, ("fees_estimated",), exit_first=True),
            "net_pnl_verification_status": _choose(
                entry, exit_, ("net_pnl_verification_status",), exit_first=True
            ),
            "stop_policy_params_source": _choose(
                entry, exit_, ("stop_policy_params_source", "shadow_policy_params_source"), exit_first=True
            ),
            "stop_policy_params_hash": _choose(
                entry, exit_, ("stop_policy_params_hash", "shadow_policy_params_hash"), exit_first=True
            ),
            "stop_policy_schema": _choose(
                entry, exit_, ("stop_policy_schema", "shadow_policy_schema"), exit_first=True
            ),
        }
        rows.append(row)
    return _normalise_time_columns(pd.DataFrame(rows))


def _directional_return(row: pd.Series) -> float:
    entry = _safe_float(row.get("entry_price"))
    exit_ = _safe_float(row.get("exit_price"))
    if not np.isfinite(entry) or not np.isfinite(exit_) or entry <= 0.0:
        return np.nan
    if str(row.get("side")).lower() == "short":
        return float((entry - exit_) / entry)
    return float((exit_ - entry) / entry)


def _artifact_exists(data_root: Path, run_id: Any, kind: str) -> bool:
    if not _is_present(run_id):
        return False
    root = data_root / "artifacts" / str(run_id)
    if kind == "model":
        return (root / "models" / "trained_state.pkl").exists() or (root / "models" / "native").exists()
    if kind == "policy":
        return (
            (root / "simple_policy_optimiser").exists()
            or (root / "policy_params").exists()
            or (root / "best_policy_params_perps.json").exists()
            or (root / "strategy_for_inference_perps.json").exists()
        )
    return root.exists()


def _policy_replay_param_keys(value: Any) -> list[str]:
    obj = _json_loads(value, {})
    if not isinstance(obj, dict):
        return []
    return sorted(k for k, v in obj.items() if _is_present(v))


def _join_predictions(trades: pd.DataFrame, ledger: pd.DataFrame) -> pd.DataFrame:
    if trades.empty or ledger.empty:
        return trades.copy()
    led = _normalise_time_columns(ledger)
    traded = led[
        led.get("portfolio_decision", pd.Series("", index=led.index))
        .astype(str)
        .str.lower()
        .isin({"trade", "traded", "accepted", "filled"})
    ].copy()
    if traded.empty:
        traded = led.copy()
    for col in ("symbol", "side", "strategy_id"):
        if col in traded.columns:
            traded[col] = traded[col].astype(str)
    out_rows: list[dict[str, Any]] = []
    for _, trade in trades.iterrows():
        cand = traded.copy()
        for col in ("symbol", "side", "strategy_id"):
            if col in cand.columns and _is_present(trade.get(col)):
                if col == "symbol":
                    target = _normalise_symbol(trade.get(col))
                    cand = cand[cand[col].map(_normalise_symbol).eq(target)]
                elif col == "side":
                    target = _normalise_side(trade.get(col))
                    cand = cand[cand[col].map(_normalise_side).eq(target)]
                else:
                    cand = cand[cand[col].astype(str).eq(str(trade.get(col)))]
        anchor = trade.get("decision_ts")
        if not isinstance(anchor, pd.Timestamp) or pd.isna(anchor):
            anchor = trade.get("entry_ts")
        matched = pd.Series(dtype=object)
        if not cand.empty and isinstance(anchor, pd.Timestamp) and pd.notna(anchor):
            decision_ts = pd.to_datetime(cand.get("decision_ts"), utc=True, errors="coerce")
            before = cand[decision_ts <= anchor + pd.Timedelta(minutes=5)].copy()
            if not before.empty:
                deltas = (pd.to_datetime(before["decision_ts"], utc=True, errors="coerce") - anchor).abs()
                matched = before.loc[deltas.idxmin()]
        combined = trade.to_dict()
        if not matched.empty:
            for col in (
                "decision_ts",
                "signal_bar_ts",
                "model_artifact_run_id",
                "policy_artifact_run_id",
                "base_model_key",
                "meta_model_key",
                "base_model_feature_count",
                "meta_model_feature_count",
                "feature_contract_hash",
                "model_feature_snapshot_hash",
                "policy_replay_params_json",
                "barrier_pct",
                "sl_mult",
                "policy_effective_barrier_pct",
                "policy_stop_price",
                "stop_policy_params_hash",
            ):
                if col in matched:
                    combined[f"ledger_{col}"] = matched.get(col)
            combined["ledger_match_found"] = True
        else:
            combined["ledger_match_found"] = False
        out_rows.append(combined)
    return _normalise_time_columns(pd.DataFrame(out_rows))


def build_bridge_report(
    *,
    ledger_path: Path,
    trade_log_path: Path,
    data_root: Path,
    run_id: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ledger = _read_table(ledger_path)
    if not ledger.empty and "policy_artifact_run_id" in ledger.columns:
        ledger = ledger[ledger["policy_artifact_run_id"].astype(str).eq(str(run_id))].copy()
    trade_log = _read_table(trade_log_path)
    trades = collapse_trade_rows(trade_log, run_id=run_id)
    bridge = _join_predictions(trades, ledger)
    if bridge.empty:
        return bridge, {
            "run_id": run_id,
            "ledger_path": str(ledger_path),
            "trade_log_path": str(trade_log_path),
            "trade_count": 0,
            "closed_trade_count": 0,
            "reason": "no_bridge_rows",
        }

    bridge["gross_return_recomputed"] = bridge.apply(_directional_return, axis=1)
    bridge["replay_param_keys"] = bridge.get("ledger_policy_replay_params_json", pd.Series("", index=bridge.index)).map(
        _policy_replay_param_keys
    )
    bridge["exact_replay_ready"] = bridge["replay_param_keys"].map(
        lambda keys: {"barrier_pct", "sl_mult"}.issubset(set(keys))
    )
    bridge["execution_bridge_ready"] = (
        pd.to_numeric(bridge.get("requested_policy_stop"), errors="coerce").notna()
        & pd.to_numeric(bridge.get("exit_vs_policy_stop_bps"), errors="coerce").notna()
    )
    bridge["model_artifact_exists"] = bridge.get(
        "ledger_model_artifact_run_id", pd.Series("", index=bridge.index)
    ).map(lambda rid: _artifact_exists(data_root, rid, "model"))
    bridge["policy_artifact_exists"] = bridge.get(
        "ledger_policy_artifact_run_id", pd.Series("", index=bridge.index)
    ).map(lambda rid: _artifact_exists(data_root, rid, "policy"))

    closed = bridge[pd.to_datetime(bridge.get("exit_ts"), utc=True, errors="coerce").notna()].copy()
    gap = pd.to_numeric(closed.get("exit_vs_policy_stop_bps"), errors="coerce")
    entry_spread_delta = pd.to_numeric(closed.get("entry_vs_expected_spread_bps"), errors="coerce")
    exit_spread_delta = pd.to_numeric(closed.get("exit_vs_expected_spread_bps"), errors="coerce")
    summary = {
        "run_id": run_id,
        "ledger_path": str(ledger_path),
        "trade_log_path": str(trade_log_path),
        "trade_count": int(len(bridge)),
        "closed_trade_count": int(len(closed)),
        "ledger_match_count": int(bridge["ledger_match_found"].sum()),
        "exact_replay_ready_count": int(bridge["exact_replay_ready"].sum()),
        "execution_bridge_ready_count": int(bridge["execution_bridge_ready"].sum()),
        "model_artifact_missing_count": int((~bridge["model_artifact_exists"]).sum()),
        "policy_artifact_missing_count": int((~bridge["policy_artifact_exists"]).sum()),
        "mean_exit_vs_policy_stop_bps": float(gap.mean()) if gap.notna().any() else np.nan,
        "max_abs_exit_vs_policy_stop_bps": float(gap.abs().max()) if gap.notna().any() else np.nan,
        "mean_entry_spread_delta_bps": float(entry_spread_delta.mean())
        if entry_spread_delta.notna().any()
        else np.nan,
        "mean_exit_spread_delta_bps": float(exit_spread_delta.mean())
        if exit_spread_delta.notna().any()
        else np.nan,
        "by_close_execution_method": (
            closed.groupby("close_execution_method", dropna=False)
            .agg(
                trades=("position_id", "count"),
                mean_exit_gap_bps=("exit_vs_policy_stop_bps", lambda s: pd.to_numeric(s, errors="coerce").mean()),
                max_abs_exit_gap_bps=(
                    "exit_vs_policy_stop_bps",
                    lambda s: pd.to_numeric(s, errors="coerce").abs().max(),
                ),
            )
            .reset_index()
            .to_dict(orient="records")
            if "close_execution_method" in closed.columns
            else []
        ),
    }
    return bridge, summary


def _write_markdown(path: Path, summary: dict[str, Any], bridge: pd.DataFrame) -> None:
    closed = bridge[pd.to_datetime(bridge.get("exit_ts"), utc=True, errors="coerce").notna()].copy()
    lines = [
        "# Live Execution Policy Bridge",
        "",
        f"- Run id: `{summary.get('run_id')}`",
        f"- Prediction ledger: `{summary.get('ledger_path')}`",
        f"- Trade log: `{summary.get('trade_log_path')}`",
        f"- Trades: {summary.get('trade_count', 0)}",
        f"- Closed trades: {summary.get('closed_trade_count', 0)}",
        f"- Ledger matches: {summary.get('ledger_match_count', 0)}",
        f"- Exact replay-ready rows: {summary.get('exact_replay_ready_count', 0)}",
        f"- Execution bridge-ready rows: {summary.get('execution_bridge_ready_count', 0)}",
        f"- Missing model artifacts by referenced run id: {summary.get('model_artifact_missing_count', 0)}",
        f"- Missing policy artifacts by referenced run id: {summary.get('policy_artifact_missing_count', 0)}",
        f"- Mean exit vs policy stop: {summary.get('mean_exit_vs_policy_stop_bps', np.nan):.3f} bps",
        f"- Max abs exit vs policy stop: {summary.get('max_abs_exit_vs_policy_stop_bps', np.nan):.3f} bps",
        "",
        "## Close Method",
        "",
    ]
    method_rows = summary.get("by_close_execution_method") or []
    if method_rows:
        lines.append("| close method | trades | mean stop gap bps | max abs stop gap bps |")
        lines.append("|---|---:|---:|---:|")
        for row in method_rows:
            lines.append(
                "| {method} | {trades} | {mean:.3f} | {maxabs:.3f} |".format(
                    method=row.get("close_execution_method"),
                    trades=row.get("trades", 0),
                    mean=_safe_float(row.get("mean_exit_gap_bps")),
                    maxabs=_safe_float(row.get("max_abs_exit_gap_bps")),
                )
            )
    else:
        lines.append("No closed rows with close method metadata.")
    if not closed.empty:
        lines.extend(["", "## Closed Trades", ""])
        show_cols = [
            "entry_ts",
            "exit_ts",
            "symbol",
            "side",
            "close_execution_method",
            "close_price_source",
            "exit_reason",
            "gross_return_recomputed",
            "net_pnl_pct",
            "exit_vs_policy_stop_bps",
            "entry_vs_expected_spread_bps",
            "exit_vs_expected_spread_bps",
            "exact_replay_ready",
            "execution_bridge_ready",
        ]
        existing = [c for c in show_cols if c in closed.columns]
        lines.append(closed[existing].to_markdown(index=False))
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--trade-log", type=Path, default=Path(DEFAULT_TRADE_LOG))
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    ledger = args.ledger or (
        data_root
        / "exchanges"
        / "krakenfutures"
        / "live_state"
        / "prediction_ledgers"
        / str(args.run_id)
        / "prediction_ledger.parquet"
    )
    out_dir = args.out_dir or data_root / "reports" / "live_execution_policy_bridge" / str(args.run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    bridge, summary = build_bridge_report(
        ledger_path=ledger,
        trade_log_path=args.trade_log,
        data_root=data_root,
        run_id=str(args.run_id),
    )
    csv_path = out_dir / "live_execution_policy_bridge.csv"
    summary_path = out_dir / "live_execution_policy_bridge_summary.json"
    md_path = out_dir / "live_execution_policy_bridge.md"
    bridge.to_csv(csv_path, index=False)
    summary_path.write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n")
    _write_markdown(md_path, summary, bridge)
    print(json.dumps(_json_safe({"csv": str(csv_path), "summary": str(summary_path), "md": str(md_path), **summary})))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
