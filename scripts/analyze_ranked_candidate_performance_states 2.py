#!/usr/bin/env python3
"""Analyze all candidate trades above a rank threshold by strategy head."""

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


HEADS = ("long_bars", "long_dist", "short_asset", "short_boll")
STATE_TO_VALUE = {"low": -1.0, "medium": 0.0, "high": 1.0}
STATE_ORDER = ("low", "medium", "high")

DEFAULT_TRAIN_BROAD = Path(
    "data_perp/artifacts/reliability_blend_native_simple_policy_replay_20260624_floor070/"
    "simple_policy_optimiser/simple_policy_candidates_broad.parquet"
)
DEFAULT_EVAL_BROAD = Path(
    "data_perp/artifacts/reliability_blend_arm_A0_anchor_only_20260625_jun15_22/"
    "simple_policy_optimiser/simple_policy_candidates_broad.parquet"
)
DEFAULT_OUTPUT_DIR = Path("data_perp/reports/ranked_candidate_state_autocorr_20260625_rank080")


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
    for head in HEADS:
        out.loc[sid.str.startswith(head + "_") | sid.eq(head)] = head
    return out


def _load_candidates(path: Path, *, source: str, rank_col: str, rank_threshold: float) -> pd.DataFrame:
    df = pd.read_parquet(path)
    rank = pd.to_numeric(_first_existing(df, (rank_col,), np.nan), errors="coerce")
    if rank.isna().all() and rank_col != "normalized_rank_score":
        rank = pd.to_numeric(_first_existing(df, ("normalized_rank_score",), np.nan), errors="coerce")
    timestamp = pd.to_datetime(_first_existing(df, ("timestamp", "candidate_timestamp"), pd.NaT), utc=True, errors="coerce")
    strategy_id = _first_existing(df, ("strategy_id", "candidate_strategy_id"), "").astype(str)
    head = _first_existing(df, ("head",), None)
    head = head.astype(str) if not head.isna().all() else _derive_head(strategy_id)
    head = head.where(~head.eq("None"), _derive_head(strategy_id))
    net_return = pd.to_numeric(_first_existing(df, ("net_return", "candidate_net_return"), np.nan), errors="coerce")
    gross_return = pd.to_numeric(_first_existing(df, ("gross_return", "candidate_gross_return"), net_return), errors="coerce")
    reason = _first_existing(df, ("simple_policy_exit_reason", "candidate_simple_policy_exit_reason"), "").astype(str).str.lower()

    out = pd.DataFrame(
        {
            "source": source,
            "timestamp": timestamp,
            "date": timestamp.dt.floor("D"),
            "head": head.astype(str),
            "strategy_id": strategy_id,
            "symbol": _first_existing(df, ("symbol", "candidate_symbol"), "").astype(str),
            "side": _first_existing(df, ("side", "candidate_side"), "").astype(str),
            "rank_score": rank.astype(float),
            "calibrated_score": pd.to_numeric(
                _first_existing(df, ("calibrated_score", "candidate_calibrated_score"), np.nan),
                errors="coerce",
            ),
            "net_return": net_return.astype(float),
            "gross_return": gross_return.astype(float),
            "cost_return": (gross_return - net_return).astype(float),
            "win": (net_return > 0.0).astype(float),
            "full_sl": reason.isin(["sl", "full_sl", "stop", "stop_loss"]).astype(float),
            "timeout": reason.eq("timeout").astype(float),
        }
    )
    out = out.loc[
        out["timestamp"].notna()
        & out["head"].isin(HEADS)
        & np.isfinite(out["rank_score"])
        & np.isfinite(out["net_return"])
        & out["rank_score"].ge(float(rank_threshold))
    ].copy()
    return out.sort_values(["timestamp", "head", "rank_score"], ascending=[True, True, False], kind="mergesort")


def _daily_panel(candidates: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        candidates.groupby(["date", "head"], observed=True)
        .agg(
            candidate_count=("timestamp", "size"),
            sum_net_return=("net_return", "sum"),
            mean_net_return=("net_return", "mean"),
            median_net_return=("net_return", "median"),
            q05_net_return=("net_return", lambda s: float(s.quantile(0.05))),
            gross_return_sum=("gross_return", "sum"),
            cost_return_sum=("cost_return", "sum"),
            win_rate=("win", "mean"),
            full_sl_rate=("full_sl", "mean"),
            timeout_rate=("timeout", "mean"),
            avg_rank_score=("rank_score", "mean"),
        )
        .reset_index()
    )
    all_dates = pd.date_range(candidates["date"].min(), candidates["date"].max(), freq="D", tz="UTC")
    idx = pd.MultiIndex.from_product([all_dates, HEADS], names=["date", "head"])
    panel = grouped.set_index(["date", "head"]).reindex(idx).reset_index()
    zero_cols = ["candidate_count", "sum_net_return", "gross_return_sum", "cost_return_sum"]
    panel[zero_cols] = panel[zero_cols].fillna(0.0)
    panel["active"] = panel["candidate_count"] > 0
    panel["cum_sum_net_return"] = panel.groupby("head", observed=True)["sum_net_return"].cumsum()
    panel["rolling_7d_sum_net_return"] = (
        panel.sort_values(["head", "date"])
        .groupby("head", observed=True)["sum_net_return"]
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
    panel["performance_state"] = pd.Series(pd.NA, index=panel.index, dtype="object")
    panel["performance_state_value"] = np.nan
    for head, g in panel.sort_values("date").groupby("head", observed=True):
        active = g.loc[g["active"]].copy()
        if active.empty:
            continue
        q33 = float(active["sum_net_return"].quantile(1.0 / 3.0))
        q67 = float(active["sum_net_return"].quantile(2.0 / 3.0))
        active["state"] = "medium"
        active.loc[active["sum_net_return"] <= q33, "state"] = "low"
        active.loc[active["sum_net_return"] >= q67, "state"] = "high"
        active["state_value"] = active["state"].map(STATE_TO_VALUE).astype(float)
        panel.loc[active.index, "performance_state"] = active["state"]
        panel.loc[active.index, "performance_state_value"] = active["state_value"]

        lag_state = active["state"].shift(1)
        transition_mask = lag_state.notna()
        same = float((active.loc[transition_mask, "state"] == lag_state.loc[transition_mask]).mean()) if transition_mask.any() else np.nan
        for prev in STATE_ORDER:
            prev_mask = transition_mask & lag_state.eq(prev)
            rec = {"head": head, "from_state": prev, "n": int(prev_mask.sum())}
            for nxt in STATE_ORDER:
                rec[f"to_{nxt}"] = float(active.loc[prev_mask, "state"].eq(nxt).mean()) if rec["n"] > 0 else np.nan
            transitions.append(rec)
        rows.append(
            {
                "head": head,
                "calendar_days": int(len(g)),
                "active_days": int(len(active)),
                "candidate_count": int(active["candidate_count"].sum()),
                "sum_net_return": float(active["sum_net_return"].sum()),
                "mean_candidate_net_return": float(
                    active["sum_net_return"].sum() / max(active["candidate_count"].sum(), 1.0)
                ),
                "mean_active_daily_sum_return": float(active["sum_net_return"].mean()),
                "q33_active_daily_sum_return": q33,
                "q67_active_daily_sum_return": q67,
                "win_rate": float(
                    np.average(active["win_rate"].fillna(0.0), weights=np.maximum(active["candidate_count"], 1.0))
                ),
                "full_sl_rate": float(
                    np.average(active["full_sl_rate"].fillna(0.0), weights=np.maximum(active["candidate_count"], 1.0))
                ),
                "avg_rank_score": float(
                    np.average(active["avg_rank_score"].fillna(0.0), weights=np.maximum(active["candidate_count"], 1.0))
                ),
                "state_same_next_active_day": same,
                "state_value_autocorr_lag1_active": _safe_autocorr(active["state_value"], 1),
                "state_value_autocorr_lag3_active": _safe_autocorr(active["state_value"], 3),
                "sum_return_autocorr_lag1_calendar": _safe_autocorr(g["sum_net_return"], 1),
                "sum_return_autocorr_lag3_calendar": _safe_autocorr(g["sum_net_return"], 3),
                "sum_return_autocorr_lag7_calendar": _safe_autocorr(g["sum_net_return"], 7),
                "low_indicator_autocorr_lag1_active": _safe_autocorr(active["state"].eq("low").astype(float), 1),
                "medium_indicator_autocorr_lag1_active": _safe_autocorr(active["state"].eq("medium").astype(float), 1),
                "high_indicator_autocorr_lag1_active": _safe_autocorr(active["state"].eq("high").astype(float), 1),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(transitions)


def _plot(panel: pd.DataFrame, output_dir: Path, rank_threshold: float) -> tuple[Path, Path]:
    colors = {
        "long_bars": "#2f6fbb",
        "long_dist": "#2c9a6b",
        "short_asset": "#c04b4b",
        "short_boll": "#8f63c7",
    }
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=False, constrained_layout=True)
    date_values = panel.loc[panel["head"].eq(HEADS[0])].sort_values("date")["date"]
    date_nums = mdates.date2num(date_values.dt.tz_convert(None))
    for head in HEADS:
        h = panel.loc[panel["head"].eq(head)].sort_values("date")
        axes[0].plot(h["date"], h["cum_sum_net_return"], label=head, color=colors[head], linewidth=2)
        axes[1].plot(h["date"], h["rolling_7d_sum_net_return"], label=head, color=colors[head], linewidth=1.8)
    axes[0].axhline(0.0, color="#333333", linewidth=0.8, alpha=0.5)
    axes[0].set_title(f"Cumulative equal-notional net return sum by head, rank >= {rank_threshold:.2f}")
    axes[0].set_ylabel("Sum net return")
    axes[0].legend(ncol=4, loc="upper left")
    axes[0].set_xlim(date_values.min(), date_values.max())
    axes[1].axhline(0.0, color="#333333", linewidth=0.8, alpha=0.5)
    axes[1].set_title("Rolling 7-day equal-notional net return sum")
    axes[1].set_ylabel("7D sum net return")
    axes[1].set_xlim(date_values.min(), date_values.max())

    state_arr = []
    for head in HEADS:
        state_arr.append(panel.loc[panel["head"].eq(head)].sort_values("date")["performance_state_value"].to_numpy(dtype=float))
    state_arr = np.asarray(state_arr, dtype=float)
    cmap = plt.matplotlib.colors.ListedColormap(["#c94f4f", "#d8d8d8", "#2e8f5b"])
    norm = plt.matplotlib.colors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    axes[2].imshow(
        state_arr,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
        extent=[date_nums[0] - 0.5, date_nums[-1] + 0.5, len(HEADS), 0],
    )
    axes[2].set_yticks(np.arange(len(HEADS)) + 0.5)
    axes[2].set_yticklabels(HEADS)
    axes[2].set_title("Active-day performance states: low / medium / high")
    axes[2].xaxis_date()
    axes[2].xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    axes[2].tick_params(axis="x", rotation=30)
    axes[2].set_xlabel("Calendar day")
    chart_path = output_dir / "ranked_candidate_performance_over_time.png"
    fig.savefig(chart_path, dpi=160)
    plt.close(fig)

    fig2, axes2 = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for ax, head in zip(axes2.ravel(), HEADS):
        h = panel.loc[panel["head"].eq(head)].sort_values("date")
        active = h.loc[h["active"]]
        ax.bar(h["date"], h["sum_net_return"], color=np.where(h["sum_net_return"] >= 0.0, "#2e8f5b", "#c94f4f"), alpha=0.72)
        ax.plot(h["date"], h["rolling_7d_sum_net_return"], color="#222222", linewidth=1.5)
        ax.scatter(
            active["date"],
            active["sum_net_return"],
            c=active["performance_state_value"],
            cmap=cmap,
            norm=norm,
            s=28,
            edgecolors="#222222",
            linewidths=0.3,
        )
        ax.axhline(0.0, color="#333333", linewidth=0.7)
        ax.set_title(head)
        ax.set_ylabel("Daily sum net return")
    detail_path = output_dir / "ranked_candidate_daily_states.png"
    fig2.savefig(detail_path, dpi=160)
    plt.close(fig2)
    return chart_path, detail_path


def _write_report(
    output_dir: Path,
    candidates: pd.DataFrame,
    metrics: pd.DataFrame,
    transitions: pd.DataFrame,
    chart_path: Path,
    detail_path: Path,
    rank_col: str,
    rank_threshold: float,
) -> None:
    lines = [
        "# Ranked Candidate Performance State Autocorrelation",
        "",
        "This report uses all candidate rows above the rank threshold, not only accepted portfolio trades.",
        "Performance is equal-notional net return because candidate rows do not carry live portfolio position sizing.",
        "",
        f"- Rank column: `{rank_col}`",
        f"- Rank threshold: `{rank_threshold:.4f}`",
        f"- Period: `{candidates['timestamp'].min()}` to `{candidates['timestamp'].max()}`",
        f"- Candidate rows: `{len(candidates)}`",
        f"- Chart: `{chart_path}`",
        f"- Detail chart: `{detail_path}`",
        "",
        "## State Metrics",
        "",
        metrics.to_markdown(index=False, floatfmt=".4f") if not metrics.empty else "No metrics.",
        "",
        "## Transition Probabilities",
        "",
        transitions.to_markdown(index=False, floatfmt=".4f") if not transitions.empty else "No transitions.",
        "",
    ]
    (output_dir / "ranked_candidate_state_autocorr_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-broad", type=Path, default=DEFAULT_TRAIN_BROAD)
    parser.add_argument("--eval-broad", type=Path, default=DEFAULT_EVAL_BROAD)
    parser.add_argument("--rank-col", default="normalized_rank_score")
    parser.add_argument("--rank-threshold", type=float, default=0.80)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    if args.train_broad.exists():
        parts.append(_load_candidates(args.train_broad, source="train_broad", rank_col=args.rank_col, rank_threshold=args.rank_threshold))
    if args.eval_broad.exists():
        parts.append(_load_candidates(args.eval_broad, source="jun15_22_eval_broad", rank_col=args.rank_col, rank_threshold=args.rank_threshold))
    if not parts:
        raise FileNotFoundError("No candidate inputs found.")
    candidates = pd.concat(parts, ignore_index=True)
    candidates = candidates.drop_duplicates(
        subset=["timestamp", "head", "strategy_id", "symbol", "side", "net_return", "rank_score"],
        keep="last",
    ).sort_values(["timestamp", "head", "rank_score"], ascending=[True, True, False], kind="mergesort")
    panel = _daily_panel(candidates)
    metrics, transitions = _state_metrics(panel)
    chart_path, detail_path = _plot(panel, args.output_dir, args.rank_threshold)

    candidates.to_parquet(args.output_dir / "ranked_candidate_rows.parquet", index=False)
    panel.to_csv(args.output_dir / "daily_ranked_candidate_performance.csv", index=False)
    metrics.to_csv(args.output_dir / "ranked_candidate_state_autocorr_metrics.csv", index=False)
    transitions.to_csv(args.output_dir / "ranked_candidate_state_transition_matrix.csv", index=False)
    manifest = {
        "generated_by": "analyze_ranked_candidate_performance_states",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_broad": str(args.train_broad),
        "eval_broad": str(args.eval_broad),
        "rank_col": str(args.rank_col),
        "rank_threshold": float(args.rank_threshold),
        "period_start": candidates["timestamp"].min(),
        "period_end": candidates["timestamp"].max(),
        "candidate_count": int(len(candidates)),
        "outputs": {
            "chart": str(chart_path),
            "detail_chart": str(detail_path),
            "metrics": str(args.output_dir / "ranked_candidate_state_autocorr_metrics.csv"),
            "transitions": str(args.output_dir / "ranked_candidate_state_transition_matrix.csv"),
            "daily": str(args.output_dir / "daily_ranked_candidate_performance.csv"),
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2) + "\n", encoding="utf-8")
    _write_report(
        args.output_dir,
        candidates,
        metrics,
        transitions,
        chart_path,
        detail_path,
        args.rank_col,
        float(args.rank_threshold),
    )
    print(json.dumps(_json_safe(manifest), indent=2))


if __name__ == "__main__":
    main()
