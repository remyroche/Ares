#!/usr/bin/env python3
"""Audit simple-policy OOS execution realism against live inference diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd


DEFAULT_RUN_ID = "20260525_010004_nopenalty"
DEFAULT_ARTIFACT_ROOT = (
    "data_perp/exchanges/krakenfutures/artifacts/20260525_010004_nopenalty"
)
DEFAULT_LEDGER = "data_perp/exchanges/krakenfutures/live_state/prediction_ledger.parquet"
DEFAULT_OUTPUT_DIR = (
    "extreme_price_movements/reports/inference_mismatch_investigation"
)


def _short_strategy(strategy_id: Any) -> str:
    text = str(strategy_id or "")
    if text.startswith("long_dist"):
        return "long_dist"
    if text.startswith("long_loc"):
        return "long_loc"
    if text.startswith("short_dist"):
        return "short_dist"
    if text.startswith("short_loc"):
        return "short_loc"
    side = "short" if text.startswith("short_") or "_short" in text else "long"
    kind = "loc" if text.startswith(f"{side}_loc") else "dist"
    return f"{side}_{kind}"


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def _load_thresholds(policy_json: Path) -> Dict[str, float]:
    obj = json.loads(policy_json.read_text())
    out: Dict[str, float] = {}
    for row in obj.get("strategies") or []:
        sid = str(row.get("strategy_id") or "")
        if not sid:
            continue
        out[sid] = _safe_float(row.get("deployment_rank_threshold"))
    return out


def _strategy_threshold_frame(thresholds: Dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "strategy_id": sid,
                "strategy_short": _short_strategy(sid),
                "deployment_rank_threshold": thr,
            }
            for sid, thr in sorted(thresholds.items())
        ]
    )


def _prepare_candidates(candidates_path: Path, thresholds: Dict[str, float]) -> pd.DataFrame:
    df = pd.read_parquet(candidates_path)
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["strategy_id"] = df["strategy_id"].astype(str)
    df["strategy_short"] = df["strategy_id"].map(_short_strategy)
    df["deployment_rank_threshold"] = df["strategy_id"].map(thresholds)
    rank_col = "auction_rank_score" if "auction_rank_score" in df.columns else "strategy_rank_pct"
    df["deployment_rank_col"] = rank_col
    df["passes_deployment_rank"] = (
        pd.to_numeric(df.get(rank_col), errors="coerce")
        >= pd.to_numeric(df["deployment_rank_threshold"], errors="coerce")
    )
    side = pd.to_numeric(df.get("side", 1.0), errors="coerce").fillna(1.0)
    entry = pd.to_numeric(df.get("entry_price"), errors="coerce")
    theoretical_entry = pd.to_numeric(df.get("theoretical_entry_price"), errors="coerce")
    exit_price = pd.to_numeric(df.get("exit_price"), errors="coerce")
    gross = pd.to_numeric(df.get("gross_return"), errors="coerce")
    net = pd.to_numeric(df.get("net_return"), errors="coerce")
    fees_bps = pd.to_numeric(df.get("fees_bps"), errors="coerce")
    fallback_friction_bps = (gross - net) * 10_000.0
    fees_bps = fees_bps.fillna(fallback_friction_bps)
    no_delay_gross = side * (exit_price / theoretical_entry.replace(0.0, np.nan) - 1.0)
    no_delay_net = no_delay_gross - (fees_bps / 10_000.0)
    # Diagnostic only: this is a same-exit price comparison, not a valid
    # no-delay policy replay. The proper no-delay replay comes from
    # execution_attribution/*.csv when available.
    df["same_exit_price_gap_gross_return"] = no_delay_gross
    df["same_exit_price_gap_net_return"] = no_delay_net
    df["same_exit_price_gap_bps"] = (gross - no_delay_gross) * 10_000.0
    df["computed_friction_drag_bps"] = (gross - net) * 10_000.0
    df["entry_gap_bps"] = pd.to_numeric(df.get("entry_gap_bps"), errors="coerce")
    df["entry_slippage_proxy_bps"] = pd.to_numeric(
        df.get("entry_slippage_proxy_bps"), errors="coerce"
    )
    df["expected_friction_bps"] = pd.to_numeric(
        df.get("expected_friction_bps"), errors="coerce"
    )
    df["orderbook_slippage_bps"] = pd.to_numeric(
        df.get("orderbook_slippage_bps"), errors="coerce"
    )
    df["slippage_bps"] = pd.to_numeric(df.get("slippage_bps"), errors="coerce")
    df["gross_return"] = gross
    df["net_return"] = net
    df["fees_bps"] = fees_bps
    df["entry_price"] = entry
    if {"timestamp", "delayed_entry_ts"}.issubset(df.columns):
        delayed = pd.to_datetime(df["delayed_entry_ts"], utc=True, errors="coerce")
        df["artifact_entry_delay_minutes"] = (
            (delayed - df["timestamp"]).dt.total_seconds() / 60.0
        )
    return df


def _candidate_delay_summary(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if "artifact_entry_delay_minutes" not in df.columns:
        return {"available": False}
    delay = pd.to_numeric(df["artifact_entry_delay_minutes"], errors="coerce")
    valid = delay.dropna()
    out["available"] = True
    out["non_null"] = int(valid.size)
    if valid.empty:
        return out
    counts = valid.round(6).value_counts().sort_index()
    out["minutes_value_counts"] = {str(float(k)): int(v) for k, v in counts.items()}
    out["min_minutes"] = float(valid.min())
    out["median_minutes"] = float(valid.median())
    out["max_minutes"] = float(valid.max())
    if "entry_execution_source" in df.columns:
        out["entry_execution_source_counts"] = (
            df["entry_execution_source"].fillna("missing").astype(str).value_counts().to_dict()
        )
    return out


def _trades_per_day(rows: pd.DataFrame) -> float:
    if rows.empty or "timestamp" not in rows:
        return 0.0
    ts = pd.to_datetime(rows["timestamp"], utc=True, errors="coerce").dropna()
    if ts.empty:
        return 0.0
    days = max((ts.max() - ts.min()).total_seconds() / 86400.0, 1.0)
    return float(len(rows) / days)


def _metrics(rows: pd.DataFrame, *, scope: str, strategy: str = "global") -> Dict[str, Any]:
    out: Dict[str, Any] = {"scope": scope, "strategy_short": strategy, "n": int(len(rows))}
    out["trades_per_day"] = _trades_per_day(rows)
    if rows.empty:
        return out
    gross = pd.to_numeric(rows["gross_return"], errors="coerce")
    net = pd.to_numeric(rows["net_return"], errors="coerce")
    same_exit_gap_gross = pd.to_numeric(rows["same_exit_price_gap_gross_return"], errors="coerce")
    same_exit_gap_net = pd.to_numeric(rows["same_exit_price_gap_net_return"], errors="coerce")
    out.update(
        {
            "gross_hit_rate": float((gross > 0.0).mean()),
            "net_hit_rate": float((net > 0.0).mean()),
            "same_exit_price_gap_gross_hit_rate": float((same_exit_gap_gross > 0.0).mean()),
            "same_exit_price_gap_net_hit_rate": float((same_exit_gap_net > 0.0).mean()),
            "gross_bps_mean": float(gross.mean() * 10_000.0),
            "gross_bps_median": float(gross.median() * 10_000.0),
            "net_bps_mean": float(net.mean() * 10_000.0),
            "net_bps_median": float(net.median() * 10_000.0),
            "same_exit_price_gap_gross_bps_mean": float(same_exit_gap_gross.mean() * 10_000.0),
            "same_exit_price_gap_net_bps_mean": float(same_exit_gap_net.mean() * 10_000.0),
            "same_exit_price_gap_bps_mean": float(
                pd.to_numeric(rows["same_exit_price_gap_bps"], errors="coerce").mean()
            ),
            "same_exit_price_gap_bps_median": float(
                pd.to_numeric(rows["same_exit_price_gap_bps"], errors="coerce").median()
            ),
            "friction_drag_bps_mean": float(
                pd.to_numeric(rows["computed_friction_drag_bps"], errors="coerce").mean()
            ),
            "fees_bps_mean": float(pd.to_numeric(rows["fees_bps"], errors="coerce").mean()),
            "slippage_bps_mean": float(
                pd.to_numeric(rows["slippage_bps"], errors="coerce").mean()
            ),
            "orderbook_slippage_bps_mean": float(
                pd.to_numeric(rows["orderbook_slippage_bps"], errors="coerce").mean()
            ),
            "entry_gap_bps_mean": float(
                pd.to_numeric(rows["entry_gap_bps"], errors="coerce").mean()
            ),
            "entry_gap_bps_p90": float(
                pd.to_numeric(rows["entry_gap_bps"], errors="coerce").quantile(0.90)
            ),
            "entry_slippage_proxy_bps_mean": float(
                pd.to_numeric(rows["entry_slippage_proxy_bps"], errors="coerce").mean()
            ),
            "expected_friction_bps_mean": float(
                pd.to_numeric(rows["expected_friction_bps"], errors="coerce").mean()
            ),
            "delayed_1m_fill_rate": float(
                (
                    rows.get("entry_execution_source", pd.Series(index=rows.index, dtype=object))
                    == "delayed_1m_intraminute_proxy"
                ).mean()
            ),
        }
    )
    return out


def _group_metrics(df: pd.DataFrame, *, scope: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = [_metrics(df, scope=scope, strategy="global")]
    for strategy, sub in df.groupby("strategy_short", sort=True):
        rows.append(_metrics(sub, scope=scope, strategy=str(strategy)))
    return pd.DataFrame(rows)


def _group_metrics_by_strategy_id(df: pd.DataFrame, *, scope: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for strategy_id, sub in df.groupby("strategy_id", sort=True):
        row = _metrics(sub, scope=scope, strategy=str(strategy_id))
        row["strategy_id"] = str(strategy_id)
        row["strategy_short"] = _short_strategy(strategy_id)
        rows.append(row)
    return pd.DataFrame(rows)


def _summarise_numeric(series: pd.Series, prefix: str) -> Dict[str, Any]:
    s = pd.to_numeric(series, errors="coerce")
    out = {f"{prefix}_non_null": int(s.notna().sum())}
    if s.notna().any():
        out[f"{prefix}_mean"] = float(s.mean())
        out[f"{prefix}_median"] = float(s.median())
        out[f"{prefix}_p90"] = float(s.quantile(0.90))
    return out


def _delay_window_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    groups: List[tuple[str, str, pd.DataFrame]] = [("global", "global", df)]
    groups.extend(
        (str(strategy_id), _short_strategy(strategy_id), sub)
        for strategy_id, sub in df.groupby("strategy_id", sort=True)
    )
    for strategy_id, strategy_short, sub in groups:
        if sub.empty:
            continue
        source = sub.get("entry_execution_source", pd.Series(index=sub.index, dtype=object))
        row: Dict[str, Any] = {
            "strategy_id": strategy_id,
            "strategy_short": strategy_short,
            "n": int(len(sub)),
            "delayed_1m_rows": int((source == "delayed_1m_intraminute_proxy").sum()),
            "theoretical_15m_open_rows": int((source == "theoretical_15m_open").sum()),
            "delayed_1m_fill_rate": float((source == "delayed_1m_intraminute_proxy").mean()),
        }
        row.update(_summarise_numeric(sub.get("entry_delay_minutes"), "entry_delay_minutes"))
        row.update(_summarise_numeric(sub.get("delay_window_candle_count"), "delay_window_candle_count"))
        row.update(_summarise_numeric(sub.get("delay_window_range_bps"), "delay_window_range_bps"))
        row.update(_summarise_numeric(sub.get("entry_gap_bps"), "entry_gap_bps"))
        row.update(_summarise_numeric(sub.get("entry_slippage_proxy_bps"), "entry_slippage_proxy_bps"))
        row.update(_summarise_numeric(sub.get("delay_close_gap_bps"), "delay_close_gap_bps"))
        row.update(_summarise_numeric(sub.get("delay_max_adverse_bps"), "delay_max_adverse_bps"))
        row.update(_summarise_numeric(sub.get("delay_max_favorable_bps"), "delay_max_favorable_bps"))
        row.update(_summarise_numeric(sub.get("liquidity_capacity_weight"), "liquidity_capacity_weight"))
        rows.append(row)
    return pd.DataFrame(rows)


def _load_execution_attribution(policy_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    attr_dir = policy_dir / "execution_attribution"
    global_path = attr_dir / "global_summary.csv"
    strategy_path = attr_dir / "per_strategy.csv"
    global_df = pd.read_csv(global_path) if global_path.exists() else pd.DataFrame()
    strategy_df = pd.read_csv(strategy_path) if strategy_path.exists() else pd.DataFrame()
    if not global_df.empty:
        global_df = global_df.rename(columns={global_df.columns[0]: "strategy_short"})
        global_df["strategy_short"] = "global"
    return global_df, strategy_df


def _delay_sensitivity_from_attribution(
    global_attr: pd.DataFrame,
    strategy_attr: pd.DataFrame,
) -> pd.DataFrame:
    frames = [df for df in (global_attr, strategy_attr) if isinstance(df, pd.DataFrame) and not df.empty]
    if not frames:
        return pd.DataFrame()
    src = pd.concat(frames, ignore_index=True)
    rows: List[Dict[str, Any]] = []
    for _, row in src.iterrows():
        strategy = str(row.get("strategy_short") or "global")
        trades = _safe_float(row.get("trades"), 0.0)
        rows.append(
            {
                "strategy_short": strategy,
                "variant": "delayed_entry_net",
                "trades": trades,
                "hit_rate": _safe_float(row.get("hit_rate_net")),
                "mean_return_bps": _safe_float(row.get("mean_net_return_pct")) * 100.0,
                "gross_hit_rate": _safe_float(row.get("hit_rate_gross")),
                "gross_return_bps": _safe_float(row.get("mean_gross_return_pct")) * 100.0,
                "delay_cost_bps_mean": _safe_float(row.get("delay_cost_bps_net_mean")),
                "friction_drag_bps_mean": _safe_float(row.get("gross_to_net_friction_bps_mean")),
                "delayed_1m_fill_rate": _safe_float(row.get("delayed_1m_fill_rate")),
            }
        )
        rows.append(
            {
                "strategy_short": strategy,
                "variant": "no_delay_same_exit_net",
                "trades": trades,
                "hit_rate": _safe_float(row.get("hit_rate_no_delay_net_same_exit")),
                "mean_return_bps": _safe_float(row.get("mean_no_delay_net_same_exit_pct")) * 100.0,
                "gross_hit_rate": np.nan,
                "gross_return_bps": _safe_float(row.get("mean_no_delay_gross_same_exit_pct")) * 100.0,
                "delay_cost_bps_mean": 0.0,
                "friction_drag_bps_mean": _safe_float(row.get("gross_to_net_friction_bps_mean")),
                "delayed_1m_fill_rate": _safe_float(row.get("delayed_1m_fill_rate")),
            }
        )
    return pd.DataFrame(rows)


def _adverse_rejection_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    base_scopes = {
        "all_local_candidates": df,
        "passes_current_deployment_rank": df[df["passes_deployment_rank"]],
    }
    for scope, base in base_scopes.items():
        for threshold_bps in (0.0, 50.0, 100.0, 150.0):
            if threshold_bps <= 0:
                kept = base
            else:
                adverse = pd.to_numeric(base["entry_gap_bps"], errors="coerce")
                kept = base[(adverse.isna()) | (adverse < threshold_bps)]
            reject_rate = 1.0 - (len(kept) / max(len(base), 1))
            for item in _group_metrics(
                kept, scope=f"{scope}:reject_adverse_gap_ge_{threshold_bps:.0f}bps"
            ).to_dict("records"):
                item["adverse_reject_threshold_bps"] = threshold_bps
                item["reject_rate"] = float(reject_rate)
                rows.append(item)
    return pd.DataFrame(rows)


def _extra_cost_sensitivity(df: pd.DataFrame, *, label: str) -> pd.DataFrame:
    base = df[df["passes_deployment_rank"]].copy()
    rows: List[Dict[str, Any]] = []
    for extra_bps in (0.0, 5.0, 10.0, 20.0, 50.0, 100.0):
        adjusted = base.copy()
        adjusted["net_return"] = pd.to_numeric(adjusted["net_return"], errors="coerce") - (
            extra_bps / 10_000.0
        )
        for item in _group_metrics(adjusted, scope=f"passes_current_deployment_rank:{label}_{extra_bps:.0f}bps").to_dict("records"):
            item["extra_cost_bps"] = extra_bps
            item["cost_label"] = label
            rows.append(item)
    return pd.DataFrame(rows)


def _live_ledger_summary(ledger_path: Path) -> Dict[str, Any]:
    if not ledger_path.exists():
        return {"ledger_exists": False, "path": str(ledger_path)}
    df = pd.read_parquet(ledger_path)
    out: Dict[str, Any] = {
        "ledger_exists": True,
        "path": str(ledger_path),
        "rows": int(len(df)),
        "traded_rows": int(pd.Series(df.get("was_traded", False)).fillna(False).astype(bool).sum()),
    }
    if "portfolio_decision" in df.columns:
        out["portfolio_decisions"] = (
            df["portfolio_decision"].fillna("missing").astype(str).value_counts().to_dict()
        )
    if "portfolio_reject_reason" in df.columns:
        out["portfolio_reject_reasons"] = (
            df["portfolio_reject_reason"].fillna("missing").astype(str).value_counts().to_dict()
        )
    for col in (
        "signal_to_entry_seconds",
        "decision_to_entry_seconds",
        "entry_delay_adverse_bps",
        "expected_total_entry_friction_bps",
        "spread_bps",
        "ticker_spread_bps",
        "expected_fill_slippage_bps",
        "slippage_bps",
        "realized_fee_bps",
    ):
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        out[f"{col}_non_null"] = int(s.notna().sum())
        if s.notna().any():
            out[f"{col}_mean"] = float(s.mean())
            out[f"{col}_median"] = float(s.median())
            out[f"{col}_p90"] = float(s.quantile(0.90))
    if {"decision_ts", "signal_bar_ts"}.issubset(df.columns):
        decision = pd.to_datetime(df["decision_ts"], utc=True, errors="coerce")
        signal = pd.to_datetime(df["signal_bar_ts"], utc=True, errors="coerce")
        lag = (decision - signal).dt.total_seconds()
        out["decision_minus_signal_bar_seconds_non_null"] = int(lag.notna().sum())
        if lag.notna().any():
            out["decision_minus_signal_bar_seconds_mean"] = float(lag.mean())
            out["decision_minus_signal_bar_seconds_median"] = float(lag.median())
            out["decision_minus_signal_bar_seconds_p90"] = float(lag.quantile(0.90))
    return out


def _write_markdown(
    path: Path,
    *,
    run_id: str,
    candidate_path: Path,
    policy_path: Path,
    ledger_summary: Dict[str, Any],
    global_metrics: pd.DataFrame,
    delay_sensitivity: pd.DataFrame,
    thresholds: pd.DataFrame,
    candidate_delay_summary: Dict[str, Any],
    strategy_metrics: pd.DataFrame,
    delay_window_summary: pd.DataFrame,
) -> None:
    global_rows = global_metrics[global_metrics["strategy_short"].eq("global")]
    latest = global_rows.set_index("scope").to_dict("index")
    lines = [
        "# OOS vs Inference Execution Reconciliation",
        "",
        f"Run: `{run_id}`",
        "",
        f"Candidate source: `{candidate_path}`",
        f"Policy params: `{policy_path}`",
        "",
        "## Deployment Thresholds",
        "",
        thresholds.to_markdown(index=False),
        "",
        "## Candidate Artifact Entry Delay",
        "",
        "```json",
        json.dumps(candidate_delay_summary, indent=2, sort_keys=True, default=str),
        "```",
        "",
        "## OOS Candidate Execution Breakdown",
        "",
    ]
    for scope, row in latest.items():
        lines.append(f"- `{scope}`: n={int(row.get('n', 0))}, net hit={row.get('net_hit_rate', np.nan):.3f}, gross hit={row.get('gross_hit_rate', np.nan):.3f}, mean net={row.get('net_bps_mean', np.nan):.1f} bps, mean gross={row.get('gross_bps_mean', np.nan):.1f} bps, same-exit price-gap net={row.get('same_exit_price_gap_net_bps_mean', np.nan):.1f} bps, same-exit price-gap effect={row.get('same_exit_price_gap_bps_mean', np.nan):.2f} bps, friction={row.get('friction_drag_bps_mean', np.nan):.2f} bps.")
    if isinstance(delay_sensitivity, pd.DataFrame) and not delay_sensitivity.empty:
        lines.extend(
            [
                "",
                "## Proper Delay Sensitivity",
                "",
                delay_sensitivity.to_markdown(index=False),
            ]
        )
    if isinstance(strategy_metrics, pd.DataFrame) and not strategy_metrics.empty:
        cols = [
            "strategy_id",
            "n",
            "net_hit_rate",
            "gross_hit_rate",
            "net_bps_mean",
            "gross_bps_mean",
            "friction_drag_bps_mean",
            "delayed_1m_fill_rate",
        ]
        present = [c for c in cols if c in strategy_metrics.columns]
        lines.extend(
            [
                "",
                "## Per Strategy-ID Candidate Metrics",
                "",
                strategy_metrics[present].to_markdown(index=False),
            ]
        )
    if isinstance(delay_window_summary, pd.DataFrame) and not delay_window_summary.empty:
        cols = [
            "strategy_id",
            "n",
            "delayed_1m_rows",
            "theoretical_15m_open_rows",
            "delay_window_candle_count_median",
            "entry_gap_bps_mean",
            "entry_slippage_proxy_bps_mean",
            "delay_max_adverse_bps_mean",
            "delay_max_favorable_bps_mean",
            "liquidity_capacity_weight_mean",
        ]
        present = [c for c in cols if c in delay_window_summary.columns]
        lines.extend(
            [
                "",
                "## Delay Window and Liquidity Summary",
                "",
                delay_window_summary[present].to_markdown(index=False),
            ]
        )
    lines.extend(
        [
            "",
            "## Live Ledger Coverage",
            "",
            "```json",
            json.dumps(ledger_summary, indent=2, sort_keys=True, default=str),
            "```",
            "",
            "## Live Market/Stop Order Contract",
            "",
            "- Live entries refuse to place an unprotected order unless exact simple-policy stop params and barrier context are loaded for the strategy.",
            "- In live mode, the entry order may be forced to `market`; the execution path extracts the realized exchange fill and stores it as `entry_price`/`realized_entry_price`.",
            "- Theoretical/policy/ohlcv entry prices are retained separately as audit fields (`theoretical_entry_price`, `policy_entry_price`, `ohlcv_entry_price`) and used to compute `entry_delay_adverse_bps` and entry-price deltas.",
            "- Initial STOP_LOSS and trailing/replace decisions in `simple_policy_stop.py` use the live position state's `entry_price`, which is the realized fill for live entries. This avoids stops that are accidentally too close to a worse live fill, but means optimiser replay assumptions must be tuned to match live fill distributions.",
            "- Position monitoring can classify rejected protective stops through `trigger_price_rejected` or `order_rejected`; the prediction ledger only contains portfolio-level rejection reasons, so exchange-level rejection counts still require trade-executor/order logs.",
            "",
            "## Findings",
            "",
            "- OOS simple-policy candidates contain delayed-entry gross/net returns, theoretical entry, delayed entry, entry-gap, expected friction, fee, slippage, and orderbook-slippage fields.",
            "- The candidate artifact delay summary above is measured directly from `delayed_entry_ts - timestamp`; if it differs from the current code default, the artifact must be regenerated before treating its policy metrics as current-code evidence.",
            "- The proper no-delay-vs-delayed comparison is sourced from `execution_attribution/global_summary.csv` and `execution_attribution/per_strategy.csv`. The candidate-table same-exit price-gap columns are diagnostic only and must not be interpreted as a valid no-delay policy replay.",
            "- The final candidate parquet contains the t+10 delay-window fields (`delay_close_gap_bps`, `delay_max_adverse_bps`, `delay_max_favorable_bps`, `delay_window_range_bps`, and `delay_window_candle_count`); the delay-window summary above is computed directly from those fields.",
            "- Live ledger currently has sparse realized entry timing: `signal_to_entry_seconds` and `decision_to_entry_seconds` are mostly absent for untraded rows. That is acceptable for rejected candidates but means execution-delay realism must be evaluated on traded rows plus trade logs, not solely the prediction ledger.",
            "- Rejected order analysis is represented through portfolio/liquidity rejection reasons. Exchange-level rejected market/stop order counts require trade-executor logs or exchange order history, which are not fully represented in `prediction_ledger.parquet`.",
        ],
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--artifact-root", default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--ledger", default=DEFAULT_LEDGER)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    artifact_root = Path(args.artifact_root)
    policy_dir = artifact_root / "simple_policy_optimiser"
    candidates_path = policy_dir / "simple_policy_candidates.parquet"
    policy_path = policy_dir / "deployment" / "best_policy_params_perps.json"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    thresholds = _load_thresholds(policy_path)
    threshold_frame = _strategy_threshold_frame(thresholds)
    candidates = _prepare_candidates(candidates_path, thresholds)
    global_attr, strategy_attr = _load_execution_attribution(policy_dir)
    candidate_delay = _candidate_delay_summary(candidates)

    global_metrics = pd.concat(
        [
            _group_metrics(candidates, scope="all_local_candidates"),
            _group_metrics(
                candidates[candidates["passes_deployment_rank"]],
                scope="passes_current_deployment_rank",
            ),
        ],
        ignore_index=True,
    )
    strategy_metrics = pd.concat(
        [
            _group_metrics_by_strategy_id(candidates, scope="all_local_candidates"),
            _group_metrics_by_strategy_id(
                candidates[candidates["passes_deployment_rank"]],
                scope="passes_current_deployment_rank",
            ),
        ],
        ignore_index=True,
    )
    delay_windows = _delay_window_summary(candidates)
    delay_sens = _delay_sensitivity_from_attribution(global_attr, strategy_attr)
    adverse_sens = _adverse_rejection_sensitivity(candidates)
    slippage_sens = _extra_cost_sensitivity(candidates, label="extra_slippage")
    spread_sens = _extra_cost_sensitivity(candidates, label="extra_spread")
    ledger_summary = _live_ledger_summary(Path(args.ledger))

    threshold_frame.to_csv(out_dir / "execution_assumption_matrix.csv", index=False)
    global_metrics.to_csv(out_dir / "execution_realism_oos_breakdown.csv", index=False)
    strategy_metrics.to_csv(out_dir / "execution_realism_by_strategy_id.csv", index=False)
    delay_windows.to_csv(out_dir / "delay_window_summary.csv", index=False)
    delay_sens.to_csv(out_dir / "execution_delay_sensitivity.csv", index=False)
    adverse_sens.to_csv(out_dir / "adverse_entry_gap_rejection_sensitivity.csv", index=False)
    slippage_sens.to_csv(out_dir / "slippage_sensitivity.csv", index=False)
    spread_sens.to_csv(out_dir / "spread_cost_sensitivity.csv", index=False)
    (out_dir / "live_execution_ledger_summary.json").write_text(
        json.dumps(ledger_summary, indent=2, sort_keys=True, default=str) + "\n"
    )
    (out_dir / "candidate_entry_delay_summary.json").write_text(
        json.dumps(candidate_delay, indent=2, sort_keys=True, default=str) + "\n"
    )
    _write_markdown(
        out_dir / "oos_vs_inference_execution_reconciliation.md",
        run_id=args.run_id,
        candidate_path=candidates_path,
        policy_path=policy_path,
        ledger_summary=ledger_summary,
        global_metrics=global_metrics,
        delay_sensitivity=delay_sens,
        thresholds=threshold_frame,
        candidate_delay_summary=candidate_delay,
        strategy_metrics=strategy_metrics[
            strategy_metrics["scope"].eq("passes_current_deployment_rank")
        ],
        delay_window_summary=delay_windows,
    )
    _write_markdown(
        out_dir / "fill_quality_report.md",
        run_id=args.run_id,
        candidate_path=candidates_path,
        policy_path=policy_path,
        ledger_summary=ledger_summary,
        global_metrics=global_metrics,
        delay_sensitivity=delay_sens,
        thresholds=threshold_frame,
        candidate_delay_summary=candidate_delay,
        strategy_metrics=strategy_metrics[
            strategy_metrics["scope"].eq("passes_current_deployment_rank")
        ],
        delay_window_summary=delay_windows,
    )
    print(f"Wrote execution realism audit to {out_dir}")


if __name__ == "__main__":
    main()
