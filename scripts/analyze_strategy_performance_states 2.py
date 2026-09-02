#!/usr/bin/env python3
"""Chart per-strategy policy performance and performance-state persistence."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HEAD_PREFIXES = ("long_bars", "long_dist", "short_asset", "short_boll")
STATE_TO_VALUE = {"low": -1.0, "medium": 0.0, "high": 1.0}
STATE_ORDER = ("low", "medium", "high")


DEFAULT_HISTORICAL = Path(
    "data_perp/reports/reliability_blend_component_arm_portfolio_ablation_20260625/"
    "A0_anchor_only/historical_refit/refit_bar4_strategy_bar2_accepted_trades.parquet"
)
DEFAULT_EVAL = Path(
    "data_perp/reports/recent_head_activation_optuna_20260625_v9_baseline_relative_defensive/"
    "baseline_eval_accepted_trades.parquet"
)
DEFAULT_OUTPUT_DIR = Path("data_perp/reports/strategy_performance_state_autocorr_20260625")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _first_existing(df: pd.DataFrame, names: tuple[str, ...], default: Any = np.nan) -> pd.Series:
    for name in names:
        if name in df.columns:
            return df[name]
    return pd.Series(default, index=df.index)


def _derive_head(strategy_id: pd.Series) -> pd.Series:
    sid = strategy_id.astype(str)
    out = pd.Series("unknown", index=sid.index, dtype=object)
    for head in HEAD_PREFIXES:
        out.loc[sid.str.startswith(head + "_") | sid.eq(head)] = head
    return out


def _normalise_accepted(path: Path, *, source: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    timestamp = pd.to_datetime(_first_existing(df, ("timestamp", "candidate_timestamp")), utc=True, errors="coerce")
    strategy_id = _first_existing(df, ("strategy_id", "candidate_strategy_id"), "")
    head = _first_existing(df, ("head",), None)
    head = head.astype(str) if not head.isna().all() else _derive_head(strategy_id)
    head = head.where(~head.eq("None"), _derive_head(strategy_id))

    position_size = pd.to_numeric(_first_existing(df, ("position_size", "candidate_position_size"), 1.0), errors="coerce")
    net_return = pd.to_numeric(_first_existing(df, ("net_return", "candidate_net_return"), 0.0), errors="coerce")
    gross_return = pd.to_numeric(
        _first_existing(df, ("gross_return", "candidate_gross_return"), net_return),
        errors="coerce",
    )
    net_pnl = pd.to_numeric(_first_existing(df, ("net_pnl",), np.nan), errors="coerce")
    net_pnl = net_pnl.where(net_pnl.notna(), net_return * position_size)
    gross_pnl = pd.to_numeric(_first_existing(df, ("gross_pnl",), np.nan), errors="coerce")
    gross_pnl = gross_pnl.where(gross_pnl.notna(), gross_return * position_size)
    cost_pnl = pd.to_numeric(_first_existing(df, ("cost_pnl",), np.nan), errors="coerce")
    cost_pnl = cost_pnl.where(cost_pnl.notna(), gross_pnl - net_pnl)
    exit_reason = _first_existing(df, ("simple_policy_exit_reason", "candidate_simple_policy_exit_reason"), "").astype(str)

    out = pd.DataFrame(
        {
            "source": source,
            "timestamp": timestamp,
            "date": timestamp.dt.floor("D"),
            "head": head.astype(str),
            "strategy_id": strategy_id.astype(str),
            "symbol": _first_existing(df, ("symbol", "candidate_symbol"), "").astype(str),
            "side": _first_existing(df, ("side", "candidate_side"), "").astype(str),
            "position_size": position_size.astype(float),
            "net_return": net_return.astype(float),
            "gross_return": gross_return.astype(float),
            "net_pnl": net_pnl.astype(float),
            "gross_pnl": gross_pnl.astype(float),
            "cost_pnl": cost_pnl.astype(float),
            "win": (net_return > 0.0).astype(float),
            "full_sl": exit_reason.str.lower().isin(["sl", "full_sl", "stop", "stop_loss"]).astype(float),
            "timeout": exit_reason.str.lower().eq("timeout").astype(float),
        }
    )
    out = out.loc[out["timestamp"].notna() & out["head"].isin(HEAD_PREFIXES)].copy()
    return out.sort_values(["timestamp", "head"], kind="mergesort").reset_index(drop=True)


def _daily_panel(trades: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        trades.groupby(["date", "head"], observed=True)
        .agg(
            trade_count=("timestamp", "size"),
            net_pnl=("net_pnl", "sum"),
            gross_pnl=("gross_pnl", "sum"),
            cost_pnl=("cost_pnl", "sum"),
            win_rate=("win", "mean"),
            full_sl_rate=("full_sl", "mean"),
            timeout_rate=("timeout", "mean"),
            mean_net_return=("net_return", "mean"),
        )
        .reset_index()
    )
    all_dates = pd.date_range(trades["date"].min(), trades["date"].max(), freq="D", tz="UTC")
    idx = pd.MultiIndex.from_product([all_dates, HEAD_PREFIXES], names=["date", "head"])
    panel = grouped.set_index(["date", "head"]).reindex(idx).reset_index()
    fill_zero = ["trade_count", "net_pnl", "gross_pnl", "cost_pnl"]
    panel[fill_zero] = panel[fill_zero].fillna(0.0)
    panel["active"] = panel["trade_count"] > 0
    panel["cum_net_pnl"] = panel.groupby("head", observed=True)["net_pnl"].cumsum()
    panel["rolling_7d_net_pnl"] = (
        panel.sort_values(["head", "date"])
        .groupby("head", observed=True)["net_pnl"]
        .transform(lambda s: s.rolling(7, min_periods=1).sum())
    )
    return panel


def _safe_autocorr(values: pd.Series, lag: int) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if len(values) <= lag + 2:
        return np.nan
    if float(values.std(ddof=0)) <= 1e-12:
        return np.nan
    return float(values.autocorr(lag=lag))


def _state_metrics(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    for head, g in panel.sort_values("date").groupby("head", observed=True):
        full = g.copy()
        active = full.loc[full["active"]].copy()
        if active.empty:
            continue
        q33 = float(active["net_pnl"].quantile(1.0 / 3.0))
        q67 = float(active["net_pnl"].quantile(2.0 / 3.0))
        state = pd.Series("medium", index=active.index, dtype=object)
        state.loc[active["net_pnl"] <= q33] = "low"
        state.loc[active["net_pnl"] >= q67] = "high"
        active["state"] = state
        active["state_value"] = active["state"].map(STATE_TO_VALUE).astype(float)

        lag_state = active["state"].shift(1)
        transition_mask = lag_state.notna()
        same = float((active.loc[transition_mask, "state"] == lag_state.loc[transition_mask]).mean()) if transition_mask.any() else np.nan
        for prev_state in STATE_ORDER:
            prev_mask = transition_mask & lag_state.eq(prev_state)
            denom = int(prev_mask.sum())
            rec = {"head": head, "from_state": prev_state, "n": denom}
            for next_state in STATE_ORDER:
                rec[f"to_{next_state}"] = (
                    float(active.loc[prev_mask, "state"].eq(next_state).mean()) if denom > 0 else np.nan
                )
            transitions.append(rec)

        full_state_value = pd.Series(np.nan, index=full.index, dtype=float)
        full_state_value.loc[active.index] = active["state_value"]
        full["state_value"] = full_state_value
        rows.append(
            {
                "head": head,
                "calendar_days": int(len(full)),
                "active_days": int(len(active)),
                "trade_count": int(active["trade_count"].sum()),
                "net_pnl": float(active["net_pnl"].sum()),
                "mean_active_daily_pnl": float(active["net_pnl"].mean()),
                "q33_active_daily_pnl": q33,
                "q67_active_daily_pnl": q67,
                "low_state_share": float(active["state"].eq("low").mean()),
                "medium_state_share": float(active["state"].eq("medium").mean()),
                "high_state_share": float(active["state"].eq("high").mean()),
                "state_same_next_active_day": same,
                "state_value_autocorr_lag1_active": _safe_autocorr(active["state_value"], 1),
                "state_value_autocorr_lag3_active": _safe_autocorr(active["state_value"], 3),
                "daily_pnl_autocorr_lag1_calendar": _safe_autocorr(full["net_pnl"], 1),
                "daily_pnl_autocorr_lag3_calendar": _safe_autocorr(full["net_pnl"], 3),
                "daily_pnl_autocorr_lag7_calendar": _safe_autocorr(full["net_pnl"], 7),
                "low_indicator_autocorr_lag1_active": _safe_autocorr(active["state"].eq("low").astype(float), 1),
                "medium_indicator_autocorr_lag1_active": _safe_autocorr(active["state"].eq("medium").astype(float), 1),
                "high_indicator_autocorr_lag1_active": _safe_autocorr(active["state"].eq("high").astype(float), 1),
            }
        )
        panel.loc[active.index, "performance_state"] = active["state"]
        panel.loc[active.index, "performance_state_value"] = active["state_value"]
    return pd.DataFrame(rows), pd.DataFrame(transitions)


def _plot(panel: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    heads = list(HEAD_PREFIXES)
    colors = {
        "long_bars": "#2f6fbb",
        "long_dist": "#2c9a6b",
        "short_asset": "#c04b4b",
        "short_boll": "#8f63c7",
    }
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=False, constrained_layout=True)
    date_values = panel.loc[panel["head"].eq(heads[0])].sort_values("date")["date"]
    date_nums = mdates.date2num(date_values.dt.tz_convert(None))
    for head in heads:
        h = panel.loc[panel["head"].eq(head)].sort_values("date")
        axes[0].plot(h["date"], h["cum_net_pnl"], label=head, color=colors[head], linewidth=2)
        axes[1].plot(h["date"], h["rolling_7d_net_pnl"], label=head, color=colors[head], linewidth=1.8)
    axes[0].axhline(0.0, color="#333333", linewidth=0.8, alpha=0.5)
    axes[0].set_title("Cumulative net PnL by strategy head")
    axes[0].set_ylabel("Net PnL")
    axes[0].legend(ncol=4, loc="upper left")
    axes[0].set_xlim(date_values.min(), date_values.max())
    axes[1].axhline(0.0, color="#333333", linewidth=0.8, alpha=0.5)
    axes[1].set_title("Rolling 7-day net PnL by strategy head")
    axes[1].set_ylabel("7D net PnL")
    axes[1].set_xlim(date_values.min(), date_values.max())

    state_matrix = []
    for head in heads:
        h = panel.loc[panel["head"].eq(head)].sort_values("date")
        state_matrix.append(h["performance_state_value"].to_numpy(dtype=float))
    state_arr = np.asarray(state_matrix, dtype=float)
    cmap = plt.matplotlib.colors.ListedColormap(["#c94f4f", "#d8d8d8", "#2e8f5b"])
    norm = plt.matplotlib.colors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    axes[2].imshow(
        state_arr,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
        extent=[date_nums[0] - 0.5, date_nums[-1] + 0.5, len(heads), 0],
    )
    axes[2].set_yticks(np.arange(len(heads)) + 0.5)
    axes[2].set_yticklabels(heads)
    axes[2].set_title("Active-day performance states: low / medium / high")
    axes[2].xaxis_date()
    axes[2].xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    axes[2].tick_params(axis="x", rotation=30)
    axes[2].set_xlabel("Calendar day")
    axes[2].set_ylabel("Strategy head")
    chart_path = output_dir / "strategy_performance_over_time.png"
    fig.savefig(chart_path, dpi=160)
    plt.close(fig)

    fig2, axes2 = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    axes2 = axes2.ravel()
    for ax, head in zip(axes2, heads):
        h = panel.loc[panel["head"].eq(head)].sort_values("date")
        active = h.loc[h["active"]]
        ax.bar(h["date"], h["net_pnl"], color=np.where(h["net_pnl"] >= 0.0, "#2e8f5b", "#c94f4f"), alpha=0.75)
        ax.plot(h["date"], h["rolling_7d_net_pnl"], color="#222222", linewidth=1.5)
        ax.scatter(
            active["date"],
            active["net_pnl"],
            c=active["performance_state_value"],
            cmap=cmap,
            norm=norm,
            s=28,
            edgecolors="#222222",
            linewidths=0.3,
        )
        ax.axhline(0.0, color="#333333", linewidth=0.7)
        ax.set_title(head)
        ax.set_ylabel("Daily net PnL")
    detail_path = output_dir / "strategy_daily_pnl_states.png"
    fig2.savefig(detail_path, dpi=160)
    plt.close(fig2)
    return chart_path, detail_path


def _write_report(
    output_dir: Path,
    trades: pd.DataFrame,
    panel: pd.DataFrame,
    metrics: pd.DataFrame,
    transitions: pd.DataFrame,
    chart_path: Path,
    detail_path: Path,
) -> None:
    lines = [
        "# Strategy Performance State Autocorrelation",
        "",
        "This report uses accepted trades from the A0/T1 policy family and includes costs in net PnL.",
        "",
        f"- Period: `{trades['timestamp'].min()}` to `{trades['timestamp'].max()}`",
        f"- Trades: `{len(trades)}`",
        f"- Chart: `{chart_path}`",
        f"- Detail chart: `{detail_path}`",
        "",
        "## State Metrics",
        "",
        metrics.to_markdown(index=False, floatfmt=".4f") if not metrics.empty else "No state metrics.",
        "",
        "## Transition Probabilities",
        "",
        transitions.to_markdown(index=False, floatfmt=".4f") if not transitions.empty else "No transitions.",
        "",
    ]
    (output_dir / "strategy_performance_state_autocorr_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", type=Path, default=DEFAULT_HISTORICAL)
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    if args.historical.exists():
        parts.append(_normalise_accepted(args.historical, source="historical_refit"))
    if args.eval.exists():
        parts.append(_normalise_accepted(args.eval, source="jun15_22_eval"))
    if not parts:
        raise FileNotFoundError("No accepted-trade inputs found.")
    trades = pd.concat(parts, ignore_index=True).sort_values(["timestamp", "head"], kind="mergesort")
    trades = trades.drop_duplicates(subset=["timestamp", "head", "strategy_id", "symbol", "side", "net_return"], keep="last")
    panel = _daily_panel(trades)
    metrics, transitions = _state_metrics(panel)
    chart_path, detail_path = _plot(panel, args.output_dir)

    trades.to_parquet(args.output_dir / "normalised_accepted_trades.parquet", index=False)
    panel.to_csv(args.output_dir / "daily_strategy_performance.csv", index=False)
    metrics.to_csv(args.output_dir / "strategy_state_autocorr_metrics.csv", index=False)
    transitions.to_csv(args.output_dir / "strategy_state_transition_matrix.csv", index=False)
    manifest = {
        "generated_by": "analyze_strategy_performance_states",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "historical": str(args.historical),
        "eval": str(args.eval),
        "period_start": trades["timestamp"].min(),
        "period_end": trades["timestamp"].max(),
        "trade_count": int(len(trades)),
        "outputs": {
            "chart": str(chart_path),
            "detail_chart": str(detail_path),
            "metrics": str(args.output_dir / "strategy_state_autocorr_metrics.csv"),
            "transitions": str(args.output_dir / "strategy_state_transition_matrix.csv"),
            "daily": str(args.output_dir / "daily_strategy_performance.csv"),
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2) + "\n", encoding="utf-8")
    _write_report(args.output_dir, trades, panel, metrics, transitions, chart_path, detail_path)
    print(json.dumps(_json_safe(manifest), indent=2))


if __name__ == "__main__":
    main()
