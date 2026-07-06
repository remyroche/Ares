#!/usr/bin/env python3
"""Sweep head-specific contextual TP/SL arm combinations through portfolio replay."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.portfolio_policy_replay import (  # noqa: E402
    PortfolioPolicyParams,
    fit_hierarchical_ev_curves,
    replay_candidates,
)


ARMS = ("static", "rank_only", "performance_only", "joint_all", "independent_all")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(float(value)) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _head_name(strategy_id: str) -> str:
    text = str(strategy_id)
    if text.startswith("short_bollinger"):
        return "short_bollinger"
    parts = text.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else text


def _combo_id(mapping: Mapping[str, str]) -> str:
    order = ("long_bars", "long_dist", "short_asset", "short_bollinger")
    labels = {
        "static": "S",
        "rank_only": "R",
        "performance_only": "P",
        "joint_all": "J",
        "independent_all": "I",
    }
    return "_".join(f"{head}:{labels.get(mapping.get(head, ''), '?')}" for head in order)


def _load_arm_tables(source_dir: Path, arms: Sequence[str]) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}
    for arm in arms:
        path = source_dir / "portfolio_replay" / f"{arm}_contextual_tp_sl_candidates.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing candidate table: {path}")
        frame = pd.read_parquet(path)
        frame["strategy_id"] = frame["strategy_id"].astype(str)
        frame["head"] = frame["strategy_id"].map(_head_name)
        tables[arm] = frame
    return tables


def _accepted_period_tables(decisions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if decisions.empty or "accepted" not in decisions.columns:
        empty = pd.DataFrame()
        return empty, empty
    accepted = decisions.loc[decisions["accepted"].astype(bool)].copy()
    if accepted.empty:
        empty = pd.DataFrame()
        return empty, empty
    ts = pd.to_datetime(accepted["timestamp"], utc=True, errors="coerce")
    accepted["day"] = ts.dt.date.astype(str)
    accepted["week"] = ts.dt.to_period("W").astype(str)
    accepted["head"] = accepted["strategy_id"].map(_head_name)
    size = pd.to_numeric(accepted["position_size"], errors="coerce").fillna(0.0)
    net = pd.to_numeric(accepted["position_net_return"], errors="coerce").fillna(0.0)
    gross = pd.to_numeric(accepted["position_gross_return"], errors="coerce").fillna(0.0)
    accepted["net_pnl_amount"] = size * net
    accepted["gross_pnl_amount"] = size * gross

    daily = (
        accepted.groupby("day", as_index=False)
        .agg(
            net_pnl=("net_pnl_amount", "sum"),
            gross_pnl=("gross_pnl_amount", "sum"),
            trades=("accepted", "size"),
            hit_rate=("position_net_return", lambda x: float((pd.to_numeric(x, errors="coerce") > 0.0).mean())),
        )
        .sort_values("day")
    )
    weekly = (
        accepted.groupby("week", as_index=False)
        .agg(
            net_pnl=("net_pnl_amount", "sum"),
            gross_pnl=("gross_pnl_amount", "sum"),
            trades=("accepted", "size"),
            hit_rate=("position_net_return", lambda x: float((pd.to_numeric(x, errors="coerce") > 0.0).mean())),
        )
        .sort_values("week")
    )
    return daily, weekly


def _period_metrics(values: pd.Series, prefix: str) -> Dict[str, float]:
    arr = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_q05_pnl": 0.0,
            f"{prefix}_q10_pnl": 0.0,
            f"{prefix}_q20_pnl": 0.0,
            f"{prefix}_q35_pnl": 0.0,
            f"{prefix}_min_pnl": 0.0,
            f"{prefix}_positive_rate": 0.0,
        }
    return {
        f"{prefix}_count": int(arr.size),
        f"{prefix}_q05_pnl": float(np.nanpercentile(arr, 5)),
        f"{prefix}_q10_pnl": float(np.nanpercentile(arr, 10)),
        f"{prefix}_q20_pnl": float(np.nanpercentile(arr, 20)),
        f"{prefix}_q35_pnl": float(np.nanpercentile(arr, 35)),
        f"{prefix}_min_pnl": float(np.nanmin(arr)),
        f"{prefix}_positive_rate": float(np.nanmean(arr > 0.0)),
    }


def _concat_nonempty(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    usable = [frame for frame in frames if frame is not None and not frame.empty]
    return pd.concat(usable, ignore_index=True) if usable else pd.DataFrame()


def _dominates(left: pd.Series, right: pd.Series) -> bool:
    max_cols = ("net_pnl", "worst_week_return", "max_drawdown", "daily_q10_pnl")
    min_cols = ("full_sl_rate",)
    at_least = all(float(left[c]) >= float(right[c]) - 1e-12 for c in max_cols)
    at_least = at_least and all(float(left[c]) <= float(right[c]) + 1e-12 for c in min_cols)
    strictly = any(float(left[c]) > float(right[c]) + 1e-12 for c in max_cols)
    strictly = strictly or any(float(left[c]) < float(right[c]) - 1e-12 for c in min_cols)
    return bool(at_least and strictly)


def _add_tradeoff_scores(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    weekly_count = pd.to_numeric(out.get("weekly_count", pd.Series(0.0, index=out.index)), errors="coerce")
    net_pnl = pd.to_numeric(out.get("net_pnl", pd.Series(0.0, index=out.index)), errors="coerce")
    daily_q20 = pd.to_numeric(out.get("daily_q20_pnl", pd.Series(0.0, index=out.index)), errors="coerce")
    daily_q35 = pd.to_numeric(out.get("daily_q35_pnl", pd.Series(0.0, index=out.index)), errors="coerce")
    out["avg_week_pnl"] = np.where(weekly_count.to_numpy(dtype=float) > 0.0, net_pnl / weekly_count, 0.0)
    out["objective_avgweek_0p7dayq35_0p3dayq20"] = (
        out["avg_week_pnl"].fillna(0.0)
        + 0.7 * daily_q35.fillna(0.0)
        + 0.3 * daily_q20.fillna(0.0)
    )
    pareto = []
    for i, row in out.iterrows():
        is_dominated = False
        for j, other in out.iterrows():
            if i != j and _dominates(other, row):
                is_dominated = True
                break
        pareto.append(not is_dominated)
    out["pareto_pnl_tail"] = pareto
    for col, high in (
        ("net_pnl", True),
        ("daily_q10_pnl", True),
        ("worst_week_return", True),
        ("max_drawdown", True),
        ("full_sl_rate", False),
    ):
        vals = pd.to_numeric(out[col], errors="coerce").astype(float)
        lo = float(vals.min())
        hi = float(vals.max())
        if abs(hi - lo) <= 1e-12:
            out[f"n_{col}"] = 0.5
        elif high:
            out[f"n_{col}"] = (vals - lo) / (hi - lo)
        else:
            out[f"n_{col}"] = (hi - vals) / (hi - lo)
    out["balanced_score"] = (
        0.45 * out["n_net_pnl"]
        + 0.20 * out["n_daily_q10_pnl"]
        + 0.15 * out["n_worst_week_return"]
        + 0.15 * out["n_max_drawdown"]
        + 0.05 * out["n_full_sl_rate"]
    )
    return out


def _arm_combinations(heads: Sequence[str], arms: Sequence[str]) -> Iterable[Dict[str, str]]:
    for combo in itertools.product(arms, repeat=len(heads)):
        yield dict(zip(heads, combo))


def _load_requested_combo_ids(combo_ids: Sequence[str] | None, combo_file: Path | None) -> set[str]:
    requested = {str(combo).strip() for combo in combo_ids or [] if str(combo).strip()}
    if combo_file is not None:
        if not combo_file.exists():
            raise FileNotFoundError(f"Missing combo file: {combo_file}")
        if combo_file.suffix.lower() == ".csv":
            frame = pd.read_csv(combo_file)
            if "combo_id" not in frame.columns:
                raise ValueError(f"{combo_file} must contain a combo_id column")
            requested.update(frame["combo_id"].dropna().astype(str).str.strip())
        else:
            requested.update(line.strip() for line in combo_file.read_text().splitlines() if line.strip())
    return {combo for combo in requested if combo}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--market-mode", default="perps", choices=["spot", "perps"])
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--top-decisions", type=int, default=8)
    parser.add_argument(
        "--combo-id",
        action="append",
        default=None,
        help="Optional combo id to replay. Repeatable. When omitted, all arm combinations are replayed.",
    )
    parser.add_argument(
        "--combo-file",
        type=Path,
        default=None,
        help="Optional text file or CSV with combo_id column limiting the replay to selected combinations.",
    )
    parser.add_argument(
        "--save-accepted-decisions",
        action="store_true",
        help="Persist accepted replay decisions for requested combinations. Use with a small combo subset.",
    )
    args = parser.parse_args()

    arms = tuple(a.strip() for a in str(args.arms).split(",") if a.strip())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tables = _load_arm_tables(args.source_dir, arms)
    heads = sorted(tables[arms[0]]["head"].dropna().astype(str).unique())
    requested_combo_ids = _load_requested_combo_ids(args.combo_id, args.combo_file)
    params = PortfolioPolicyParams(global_threshold_floor=0.0)
    rows: List[Dict[str, Any]] = []
    daily_frames: List[pd.DataFrame] = []
    weekly_frames: List[pd.DataFrame] = []
    accepted_decision_frames: List[pd.DataFrame] = []
    top_payload: Dict[str, Any] = {}

    for combo_idx, mapping in enumerate(_arm_combinations(heads, arms)):
        combo_id = _combo_id(mapping)
        if requested_combo_ids and combo_id not in requested_combo_ids:
            continue
        if combo_idx and combo_idx % 50 == 0:
            print(f"completed {combo_idx} combinations", flush=True)
        frames = []
        for head, arm in mapping.items():
            source = tables[arm]
            frames.append(source.loc[source["head"].eq(head)].copy())
        candidates = (
            pd.concat(frames, ignore_index=True)
            .drop(columns=["head"], errors="ignore")
            .sort_values(["timestamp", "strategy_id", "symbol"])
            .reset_index(drop=True)
        )
        ev_curve = fit_hierarchical_ev_curves(candidates)
        decisions, equity, metrics = replay_candidates(
            candidates,
            params,
            mode="global_auction",
            ev_curve=ev_curve,
            market_mode=args.market_mode,
        )
        if args.save_accepted_decisions and "accepted" in decisions.columns:
            accepted_decisions = decisions.loc[decisions["accepted"].astype(bool)].copy()
            if not accepted_decisions.empty:
                accepted_decisions.insert(0, "combo_id", combo_id)
                for head, arm in mapping.items():
                    accepted_decisions[f"{head}_arm"] = arm
                accepted_decision_frames.append(accepted_decisions)
        daily, weekly = _accepted_period_tables(decisions)
        for frame in (daily, weekly):
            if not frame.empty:
                frame.insert(0, "combo_id", combo_id)
                for head, arm in mapping.items():
                    frame[f"{head}_arm"] = arm
        daily_frames.append(daily)
        weekly_frames.append(weekly)
        rec = {
            "combo_id": combo_id,
            "combo_index": int(combo_idx),
            **{f"{head}_arm": arm for head, arm in mapping.items()},
            "candidate_rows": int(len(candidates)),
            "candidate_start": str(pd.to_datetime(candidates["timestamp"], utc=True).min()),
            "candidate_end": str(pd.to_datetime(candidates["timestamp"], utc=True).max()),
            "objective": float(metrics.get("objective", 0.0)),
            "net_pnl": float(metrics.get("net_pnl", 0.0)),
            "gross_pnl": float(metrics.get("gross_pnl", 0.0)),
            "trade_count": int(metrics.get("trade_count", 0) or 0),
            "mean_net_return": float(metrics.get("mean_net_return_per_trade", 0.0)),
            "full_sl_rate": float(metrics.get("full_sl_rate", 0.0)),
            "timeout_rate": float(metrics.get("timeout_rate", 0.0)),
            "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
            "worst_week_return": float(metrics.get("worst_week", 0.0)),
            "strategy_concentration": float(metrics.get("strategy_concentration", 0.0)),
            "side_concentration": float(metrics.get("side_concentration", 0.0)),
        }
        rec.update(_period_metrics(daily.get("net_pnl", pd.Series(dtype=float)), "daily"))
        rec.update(_period_metrics(weekly.get("net_pnl", pd.Series(dtype=float)), "weekly"))
        rows.append(rec)

    if rows:
        summary = _add_tradeoff_scores(pd.DataFrame(rows))
        summary = summary.sort_values("balanced_score", ascending=False).reset_index(drop=True)
    else:
        summary = pd.DataFrame()
    daily_all = _concat_nonempty(daily_frames)
    weekly_all = _concat_nonempty(weekly_frames)
    summary.to_csv(args.out_dir / "head_arm_combination_summary.csv", index=False)
    daily_all.to_csv(args.out_dir / "head_arm_combination_daily.csv", index=False)
    weekly_all.to_csv(args.out_dir / "head_arm_combination_weekly.csv", index=False)
    if args.save_accepted_decisions:
        accepted_all = _concat_nonempty(accepted_decision_frames)
        if not accepted_all.empty:
            accepted_all.to_parquet(args.out_dir / "head_arm_combination_accepted_decisions.parquet", index=False)
        else:
            pd.DataFrame().to_parquet(args.out_dir / "head_arm_combination_accepted_decisions.parquet", index=False)

    lines = [
        "# Contextual TP/SL Head-Arm Combination Sweep",
        "",
        f"Source: `{args.source_dir}`",
        f"Combinations: {len(summary)}",
        f"Requested combo ids: `{len(requested_combo_ids)}`",
        "Period: full source candidate table period. Costs included. Portfolio replay uses saved contextual TP/SL outcomes.",
        "",
        "## Top Requested Objective Combinations",
        "",
    ]
    keep = [
        "combo_id",
        "objective_avgweek_0p7dayq35_0p3dayq20",
        "avg_week_pnl",
        "balanced_score",
        "pareto_pnl_tail",
        "net_pnl",
        "daily_q35_pnl",
        "daily_q20_pnl",
        "daily_q10_pnl",
        "weekly_q10_pnl",
        "worst_week_return",
        "max_drawdown",
        "full_sl_rate",
        "trade_count",
        *[f"{head}_arm" for head in heads],
    ]
    lines.append(
        summary.sort_values("objective_avgweek_0p7dayq35_0p3dayq20", ascending=False)[
            [c for c in keep if c in summary.columns]
        ]
        .head(20)
        .round(6)
        .to_markdown(index=False)
        if not summary.empty
        else "_No combinations matched the requested filters._"
    )
    lines += ["", "## Top Balanced Combinations", ""]
    lines.append(
        summary[[c for c in keep if c in summary.columns]].head(20).round(6).to_markdown(index=False)
        if not summary.empty
        else "_No combinations matched the requested filters._"
    )
    lines += ["", "## Top Net PnL Combinations", ""]
    lines.append(
        summary.sort_values("net_pnl", ascending=False)[[c for c in keep if c in summary.columns]]
        .head(20)
        .round(6)
        .to_markdown(index=False)
        if not summary.empty
        else "_No combinations matched the requested filters._"
    )
    lines += ["", "## Pareto Combinations", ""]
    lines.append(
        summary.loc[summary["pareto_pnl_tail"], [c for c in keep if c in summary.columns]]
        .sort_values("net_pnl", ascending=False)
        .head(30)
        .round(6)
        .to_markdown(index=False)
        if not summary.empty
        else "_No combinations matched the requested filters._"
    )
    (args.out_dir / "head_arm_combination_report.md").write_text("\n".join(lines) + "\n")
    payload = {
        "source_dir": str(args.source_dir),
        "out_dir": str(args.out_dir),
        "arms": list(arms),
        "heads": list(heads),
        "requested_combo_ids": sorted(requested_combo_ids),
        "combinations": int(len(summary)),
        "top_requested_objective": summary.sort_values(
            "objective_avgweek_0p7dayq35_0p3dayq20",
            ascending=False,
        )
        .head(int(args.top_decisions))
        .to_dict(orient="records"),
        "top_balanced": summary.head(int(args.top_decisions)).to_dict(orient="records"),
        "top_net_pnl": summary.sort_values("net_pnl", ascending=False).head(int(args.top_decisions)).to_dict(orient="records"),
    }
    (args.out_dir / "head_arm_combination_summary.json").write_text(
        json.dumps(_json_safe(payload), indent=2),
        encoding="utf-8",
    )
    print(json.dumps(_json_safe({"out_dir": str(args.out_dir), "combinations": len(summary)}), indent=2))


if __name__ == "__main__":
    main()
