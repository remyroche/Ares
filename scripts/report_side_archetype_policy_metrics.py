#!/usr/bin/env python3
"""Report replay coverage and execution metrics by side/archetype/time bucket."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ablate_simple_policy_exit_geometry import (  # noqa: E402
    DEFAULT_PATH_LEN,
    _load_bundles,
    _prepare_rows,
)
from scripts.validate_simple_policy_exit_geometry_walkforward import (  # noqa: E402
    DEFAULT_ARCHETYPE_COLUMN,
    _candidate_table_for_group_overrides,
    _resolve_state_column,
    _score_with_train_curve,
    _subset_bundles,
)


DEFAULT_CANDIDATES = Path(
    "data_perp/reports/replay_candidates_side_archetype_materialized_g5_meta_side_arch_local_cap15_dedup_20260705/"
    "simple_policy_candidates_with_archetypes.parquet"
)
DEFAULT_WALKFORWARD_DIR = Path(
    "data_perp/reports/simple_policy_exit_side_archetype_g5_meta_side_arch_local_cap15_dedup_cost1pct_20260705_v1"
)
DEFAULT_OUT_DIR = Path("data_perp/reports/side_archetype_policy_metrics")


def _side_label(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    text = values.astype(str).str.lower()
    return pd.Series(
        np.where(
            numeric < 0.0,
            "short",
            np.where(numeric > 0.0, "long", np.where(text.str.startswith("short"), "short", "long")),
        ),
        index=values.index,
    )


def _period_values(ts: pd.Series, period: str) -> pd.Series:
    timestamps = pd.to_datetime(ts, utc=True, errors="coerce")
    if period == "week":
        start = timestamps.dt.floor("D") - pd.to_timedelta(timestamps.dt.weekday, unit="D")
        return start.dt.strftime("%Y-%m-%d")
    if period == "month":
        return timestamps.dt.strftime("%Y-%m")
    raise ValueError(f"Unsupported period: {period}")


def _safe_ratio(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return float(values.mean()) if values.notna().any() else 0.0


def _median(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return float(values.median()) if values.notna().any() else 0.0


def _sum(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return float(values.sum()) if values.notna().any() else 0.0


def _count_reason(frame: pd.DataFrame, column: str, token: str) -> int:
    if column not in frame.columns or frame.empty:
        return 0
    text = frame[column].astype(str).str.lower()
    return int(text.eq(str(token).lower()).sum())


def _coverage_frame(
    frame: pd.DataFrame,
    *,
    count_name: str,
    period: str,
    archetype_column: str,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=["period_type", "period", "side", "policy_archetype", count_name]
        )
    work = frame.copy()
    work["period_type"] = period
    work["period"] = _period_values(work["timestamp"], period)
    work["side"] = _side_label(work["side"]) if "side" in work.columns else "unknown"
    if archetype_column in work.columns:
        work["policy_archetype"] = work[archetype_column].astype(str)
    elif "policy_archetype" in work.columns:
        work["policy_archetype"] = work["policy_archetype"].astype(str)
    else:
        work["policy_archetype"] = "missing"
    return (
        work.groupby(["period_type", "period", "side", "policy_archetype"], dropna=False)
        .size()
        .rename(count_name)
        .reset_index()
    )


def _candidate_metrics(
    candidates: pd.DataFrame,
    *,
    period: str,
) -> pd.DataFrame:
    columns = [
        "period_type",
        "period",
        "side",
        "policy_archetype",
        "executable_candidate_rows",
        "candidate_symbols",
        "candidate_mean_net_return",
        "candidate_median_net_return",
        "candidate_win_rate",
        "candidate_win_rows",
        "candidate_full_sl_rate",
        "candidate_full_sl_rows",
        "candidate_timeout_rate",
        "candidate_timeout_rows",
        "candidate_trailing_rate",
        "candidate_trailing_rows",
        "candidate_mean_holding_bars",
        "candidate_mean_rank_pct",
        "candidate_mean_score",
    ]
    if candidates.empty:
        return pd.DataFrame(columns=columns)
    work = candidates.copy()
    work["period_type"] = period
    work["period"] = _period_values(work["timestamp"], period)
    work["side"] = _side_label(work["side"])
    work["policy_archetype"] = work.get("policy_archetype", "missing").astype(str)
    rows: List[Dict[str, Any]] = []
    for keys, group in work.groupby(["period_type", "period", "side", "policy_archetype"], dropna=False):
        net = pd.to_numeric(group.get("net_return"), errors="coerce")
        full_sl_rows = _count_reason(group, "simple_policy_exit_reason", "full_sl")
        timeout_rows = _count_reason(group, "simple_policy_exit_reason", "timeout")
        trailing_rows = _count_reason(group, "simple_policy_exit_reason", "trailing")
        win_rows = int((net > 0.0).sum())
        rows.append(
            {
                "period_type": keys[0],
                "period": keys[1],
                "side": keys[2],
                "policy_archetype": keys[3],
                "executable_candidate_rows": int(len(group)),
                "candidate_symbols": int(group["symbol"].nunique()) if "symbol" in group.columns else 0,
                "candidate_mean_net_return": _mean(net),
                "candidate_median_net_return": _median(net),
                "candidate_win_rate": _safe_ratio(float(win_rows), float(net.notna().sum())),
                "candidate_win_rows": win_rows,
                "candidate_full_sl_rate": _safe_ratio(full_sl_rows, len(group)),
                "candidate_full_sl_rows": full_sl_rows,
                "candidate_timeout_rate": _safe_ratio(timeout_rows, len(group)),
                "candidate_timeout_rows": timeout_rows,
                "candidate_trailing_rate": _safe_ratio(trailing_rows, len(group)),
                "candidate_trailing_rows": trailing_rows,
                "candidate_mean_holding_bars": _mean(group.get("holding_bars", pd.Series(dtype=float))),
                "candidate_mean_rank_pct": _mean(group.get("rank_pct", pd.Series(dtype=float))),
                "candidate_mean_score": _mean(group.get("calibrated_score", pd.Series(dtype=float))),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _enrich_decisions(decisions: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty:
        return decisions.copy()
    decision = decisions.copy()
    candidate_cols = [
        "timestamp",
        "symbol",
        "side",
        "strategy_id",
        "policy_archetype",
        "rank_pct",
        "calibrated_score",
        "net_return",
        "gross_return",
        "simple_policy_exit_reason",
        "holding_bars",
    ]
    candidate_cols = [col for col in candidate_cols if col in candidates.columns]
    lookup = candidates.reset_index(drop=True)[candidate_cols].copy()
    lookup["candidate_index"] = np.arange(len(lookup), dtype=np.int64)
    return decision.merge(lookup, on="candidate_index", how="left", suffixes=("", "_candidate"))


def _accepted_metrics(
    decisions: pd.DataFrame,
    *,
    period: str,
) -> pd.DataFrame:
    columns = [
        "period_type",
        "period",
        "side",
        "policy_archetype",
        "decision_rows",
        "accepted_trades",
        "acceptance_rate",
        "accepted_symbols",
        "accepted_net_pnl",
        "accepted_gross_pnl",
        "accepted_mean_net_return",
        "accepted_median_net_return",
        "accepted_win_rate",
        "accepted_win_rows",
        "accepted_full_sl_rate",
        "accepted_full_sl_rows",
        "accepted_timeout_rate",
        "accepted_timeout_rows",
        "accepted_trailing_rate",
        "accepted_trailing_rows",
        "accepted_avg_holding_hours",
        "accepted_mean_rank_pct",
        "accepted_mean_score",
    ]
    if decisions.empty:
        return pd.DataFrame(columns=columns)
    work = decisions.copy()
    work["period_type"] = period
    work["period"] = _period_values(work["timestamp"], period)
    work["side"] = _side_label(work["side"])
    work["policy_archetype"] = work.get("policy_archetype", "missing").astype(str)
    rows: List[Dict[str, Any]] = []
    for keys, group in work.groupby(["period_type", "period", "side", "policy_archetype"], dropna=False):
        accepted = group.loc[group.get("accepted", False).astype(bool)].copy()
        net = pd.to_numeric(accepted.get("position_net_return"), errors="coerce")
        gross = pd.to_numeric(accepted.get("position_gross_return"), errors="coerce")
        size = pd.to_numeric(accepted.get("position_size"), errors="coerce")
        start = pd.to_datetime(accepted.get("timestamp"), utc=True, errors="coerce")
        end = pd.to_datetime(accepted.get("position_exit_timestamp"), utc=True, errors="coerce")
        hold_hours = (end - start).dt.total_seconds() / 3600.0
        full_sl_rows = _count_reason(accepted, "position_exit_reason", "full_sl")
        timeout_rows = _count_reason(accepted, "position_exit_reason", "timeout")
        trailing_rows = _count_reason(accepted, "position_exit_reason", "trailing")
        win_rows = int((net > 0.0).sum())
        rows.append(
            {
                "period_type": keys[0],
                "period": keys[1],
                "side": keys[2],
                "policy_archetype": keys[3],
                "decision_rows": int(len(group)),
                "accepted_trades": int(len(accepted)),
                "acceptance_rate": _safe_ratio(len(accepted), len(group)),
                "accepted_symbols": int(accepted["symbol"].nunique()) if "symbol" in accepted.columns else 0,
                "accepted_net_pnl": _sum(size * net),
                "accepted_gross_pnl": _sum(size * gross),
                "accepted_mean_net_return": _mean(net),
                "accepted_median_net_return": _median(net),
                "accepted_win_rate": _safe_ratio(float(win_rows), float(net.notna().sum())),
                "accepted_win_rows": win_rows,
                "accepted_full_sl_rate": _safe_ratio(full_sl_rows, len(accepted)),
                "accepted_full_sl_rows": full_sl_rows,
                "accepted_timeout_rate": _safe_ratio(timeout_rows, len(accepted)),
                "accepted_timeout_rows": timeout_rows,
                "accepted_trailing_rate": _safe_ratio(trailing_rows, len(accepted)),
                "accepted_trailing_rows": trailing_rows,
                "accepted_avg_holding_hours": _mean(hold_hours),
                "accepted_mean_rank_pct": _mean(accepted.get("rank_pct", pd.Series(dtype=float))),
                "accepted_mean_score": _mean(accepted.get("calibrated_score", pd.Series(dtype=float))),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _merge_metrics(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    keys = ["period_type", "period", "side", "policy_archetype"]
    out: pd.DataFrame | None = None
    for frame in frames:
        if frame.empty:
            continue
        out = frame if out is None else out.merge(frame, on=keys, how="outer")
    if out is None:
        return pd.DataFrame(columns=keys)
    for col in out.columns:
        if col not in keys:
            out[col] = out[col].fillna(0)
    return out.sort_values(keys).reset_index(drop=True)


def _load_stage_rows(path: Path, arms: Sequence[str]) -> pd.DataFrame:
    stage_path = path / "walkforward_stage_summary.csv"
    if not stage_path.exists():
        raise FileNotFoundError(stage_path)
    stage = pd.read_csv(stage_path)
    requested = set(arms)
    if "final" in requested:
        verdict_path = path / "walkforward_verdict.json"
        if verdict_path.exists():
            final_arm = str(json.loads(verdict_path.read_text()).get("final_stage_arm", ""))
            if final_arm:
                requested.add(final_arm)
    if "baseline" in requested:
        requested.add("A0_baseline")
    concrete = {arm for arm in requested if arm not in {"baseline", "final", "all"}}
    if "all" in requested:
        return stage.copy()
    return stage.loc[stage["arm"].astype(str).isin(concrete)].copy()


def _json_loads(value: Any) -> Dict[str, Any]:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return {}
    text = str(value).strip()
    if not text:
        return {}
    obj = json.loads(text)
    return obj if isinstance(obj, dict) else {}


def _build_period_report(
    *,
    raw_rows: pd.DataFrame,
    replay_rows: pd.DataFrame,
    stage_rows: pd.DataFrame,
    bundles: Sequence[Any],
    period: str,
    group_by: str,
    archetype_column: str,
    regime_column: str,
    cost_pct: float,
    market_mode: str,
    global_threshold_floor: float,
) -> pd.DataFrame:
    coverage_raw = _coverage_frame(
        raw_rows,
        count_name="raw_rows",
        period=period,
        archetype_column=archetype_column,
    )
    coverage_replay = _coverage_frame(
        replay_rows,
        count_name="replay_rows",
        period=period,
        archetype_column=archetype_column,
    )
    reports: List[pd.DataFrame] = []
    for _, stage_row in stage_rows.iterrows():
        validation_start = pd.Timestamp(stage_row["validation_start"])
        validation_end = pd.Timestamp(stage_row["validation_end"])
        train_end = pd.Timestamp(stage_row["train_end_exclusive"])
        train_bundles = _subset_bundles(bundles, end=train_end)
        validation_bundles = _subset_bundles(
            bundles,
            start=validation_start,
            end=validation_end,
        )
        if not train_bundles or not validation_bundles:
            continue
        overrides = _json_loads(stage_row.get("group_overrides_json"))
        regime_edges = _json_loads(stage_row.get("regime_edges_json"))
        if isinstance(regime_edges, dict):
            regime_edges = []
        train_candidates = _candidate_table_for_group_overrides(
            train_bundles,
            group_overrides=overrides,
            group_by=group_by,
            regime_column=regime_column,
            regime_edges=regime_edges,
            cost_pct=cost_pct,
            market_mode=market_mode,
            arm=f"{stage_row['fold']}_{stage_row['arm']}_train_report",
        )
        validation_candidates = _candidate_table_for_group_overrides(
            validation_bundles,
            group_overrides=overrides,
            group_by=group_by,
            regime_column=regime_column,
            regime_edges=regime_edges,
            cost_pct=cost_pct,
            market_mode=market_mode,
            arm=f"{stage_row['fold']}_{stage_row['arm']}_validation_report",
        )
        decisions, _equity, metrics = _score_with_train_curve(
            train_candidates=train_candidates,
            eval_candidates=validation_candidates,
            market_mode=market_mode,
            global_threshold_floor=global_threshold_floor,
        )
        enriched_decisions = _enrich_decisions(decisions, validation_candidates)
        merged = _merge_metrics(
            [
                coverage_raw,
                coverage_replay,
                _candidate_metrics(validation_candidates, period=period),
                _accepted_metrics(enriched_decisions, period=period),
            ]
        )
        if merged.empty:
            continue
        merged["fold"] = str(stage_row["fold"])
        merged["arm"] = str(stage_row["arm"])
        merged["stage"] = str(stage_row["stage"])
        merged["portfolio_objective"] = float(metrics.get("objective", 0.0))
        merged["portfolio_net_pnl"] = float(metrics.get("net_pnl", 0.0))
        merged["portfolio_trade_count"] = int(metrics.get("trade_count", 0))
        merged["path_survival_rate"] = merged.apply(
            lambda row: _safe_ratio(float(row.get("replay_rows", 0.0)), float(row.get("raw_rows", 0.0))),
            axis=1,
        )
        merged["candidate_survival_rate"] = merged.apply(
            lambda row: _safe_ratio(
                float(row.get("executable_candidate_rows", 0.0)),
                float(row.get("raw_rows", 0.0)),
            ),
            axis=1,
        )
        reports.append(merged)
    if not reports:
        return pd.DataFrame()
    columns_first = [
        "period_type",
        "period",
        "fold",
        "arm",
        "stage",
        "side",
        "policy_archetype",
    ]
    out = pd.concat(reports, ignore_index=True)
    remaining = [col for col in out.columns if col not in columns_first]
    return out[columns_first + remaining].sort_values(columns_first).reset_index(drop=True)


def _weighted_average(
    group: pd.DataFrame,
    value_col: str,
    weight_col: str,
) -> float:
    if value_col not in group.columns or weight_col not in group.columns:
        return 0.0
    values = pd.to_numeric(group[value_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    weights = pd.to_numeric(group[weight_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    mask = values.notna() & weights.notna() & weights.gt(0.0)
    if not mask.any():
        return 0.0
    return float(np.average(values[mask].to_numpy(dtype=float), weights=weights[mask].to_numpy(dtype=float)))


def _aggregate_period_report(report: pd.DataFrame) -> pd.DataFrame:
    keys = ["period_type", "period", "arm", "stage", "side", "policy_archetype"]
    if report.empty:
        return pd.DataFrame(columns=keys)
    rows: List[Dict[str, Any]] = []
    for group_key, group in report.groupby(keys, dropna=False):
        raw_rows = int(pd.to_numeric(group.get("raw_rows"), errors="coerce").max())
        replay_rows = int(pd.to_numeric(group.get("replay_rows"), errors="coerce").max())
        candidate_rows = int(pd.to_numeric(group.get("executable_candidate_rows"), errors="coerce").sum())
        accepted_trades = int(pd.to_numeric(group.get("accepted_trades"), errors="coerce").sum())
        decision_rows = int(pd.to_numeric(group.get("decision_rows"), errors="coerce").sum())
        candidate_win_rows = int(pd.to_numeric(group.get("candidate_win_rows"), errors="coerce").sum())
        candidate_full_sl_rows = int(pd.to_numeric(group.get("candidate_full_sl_rows"), errors="coerce").sum())
        candidate_timeout_rows = int(pd.to_numeric(group.get("candidate_timeout_rows"), errors="coerce").sum())
        candidate_trailing_rows = int(pd.to_numeric(group.get("candidate_trailing_rows"), errors="coerce").sum())
        accepted_win_rows = int(pd.to_numeric(group.get("accepted_win_rows"), errors="coerce").sum())
        accepted_full_sl_rows = int(pd.to_numeric(group.get("accepted_full_sl_rows"), errors="coerce").sum())
        accepted_timeout_rows = int(pd.to_numeric(group.get("accepted_timeout_rows"), errors="coerce").sum())
        accepted_trailing_rows = int(pd.to_numeric(group.get("accepted_trailing_rows"), errors="coerce").sum())
        row: Dict[str, Any] = {
            "period_type": group_key[0],
            "period": group_key[1],
            "arm": group_key[2],
            "stage": group_key[3],
            "side": group_key[4],
            "policy_archetype": group_key[5],
            "fold_rows": int(group["fold"].nunique()) if "fold" in group.columns else int(len(group)),
            "raw_rows": raw_rows,
            "replay_rows": replay_rows,
            "path_survival_rate": _safe_ratio(replay_rows, raw_rows),
            "executable_candidate_rows": candidate_rows,
            "candidate_survival_rate": _safe_ratio(candidate_rows, raw_rows),
            "candidate_symbols_sum": int(pd.to_numeric(group.get("candidate_symbols"), errors="coerce").sum()),
            "candidate_win_rows": candidate_win_rows,
            "candidate_win_rate": _safe_ratio(candidate_win_rows, candidate_rows),
            "candidate_full_sl_rows": candidate_full_sl_rows,
            "candidate_full_sl_rate": _safe_ratio(candidate_full_sl_rows, candidate_rows),
            "candidate_timeout_rows": candidate_timeout_rows,
            "candidate_timeout_rate": _safe_ratio(candidate_timeout_rows, candidate_rows),
            "candidate_trailing_rows": candidate_trailing_rows,
            "candidate_trailing_rate": _safe_ratio(candidate_trailing_rows, candidate_rows),
            "candidate_mean_net_return": _weighted_average(
                group, "candidate_mean_net_return", "executable_candidate_rows"
            ),
            "candidate_median_net_return_mean": _weighted_average(
                group, "candidate_median_net_return", "executable_candidate_rows"
            ),
            "candidate_mean_holding_bars": _weighted_average(
                group, "candidate_mean_holding_bars", "executable_candidate_rows"
            ),
            "decision_rows": decision_rows,
            "accepted_trades": accepted_trades,
            "acceptance_rate": _safe_ratio(accepted_trades, decision_rows),
            "accepted_symbols_sum": int(pd.to_numeric(group.get("accepted_symbols"), errors="coerce").sum()),
            "accepted_net_pnl": _sum(group.get("accepted_net_pnl", pd.Series(dtype=float))),
            "accepted_gross_pnl": _sum(group.get("accepted_gross_pnl", pd.Series(dtype=float))),
            "accepted_win_rows": accepted_win_rows,
            "accepted_win_rate": _safe_ratio(accepted_win_rows, accepted_trades),
            "accepted_full_sl_rows": accepted_full_sl_rows,
            "accepted_full_sl_rate": _safe_ratio(accepted_full_sl_rows, accepted_trades),
            "accepted_timeout_rows": accepted_timeout_rows,
            "accepted_timeout_rate": _safe_ratio(accepted_timeout_rows, accepted_trades),
            "accepted_trailing_rows": accepted_trailing_rows,
            "accepted_trailing_rate": _safe_ratio(accepted_trailing_rows, accepted_trades),
            "accepted_mean_net_return": _weighted_average(
                group, "accepted_mean_net_return", "accepted_trades"
            ),
            "accepted_median_net_return_mean": _weighted_average(
                group, "accepted_median_net_return", "accepted_trades"
            ),
            "accepted_avg_holding_hours": _weighted_average(
                group, "accepted_avg_holding_hours", "accepted_trades"
            ),
            "portfolio_net_pnl_sum": _sum(group.get("portfolio_net_pnl", pd.Series(dtype=float))),
            "portfolio_trade_count_sum": int(
                pd.to_numeric(group.get("portfolio_trade_count"), errors="coerce").sum()
            ),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--walkforward-dir", type=Path, default=DEFAULT_WALKFORWARD_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--data-root", default="data_perp")
    parser.add_argument("--market-mode", default="perps")
    parser.add_argument("--path-len", type=int, default=DEFAULT_PATH_LEN)
    parser.add_argument("--min-rows-per-strategy", type=int, default=5)
    parser.add_argument("--min-rank", type=float, default=0.0)
    parser.add_argument("--group-by", default="side_archetype")
    parser.add_argument("--archetype-column", default=DEFAULT_ARCHETYPE_COLUMN)
    parser.add_argument("--regime-column", default="policy_archetype")
    parser.add_argument("--global-threshold-floor", type=float, default=0.0)
    parser.add_argument("--cost-pct", type=float, default=0.005)
    parser.add_argument(
        "--download-missing-1m",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Allow the replay store to fill missing execution paths. This sets both "
            "EPM_SIMPLE_POLICY_1M_DOWNLOAD and EPM_SIMPLE_POLICY_15M_DOWNLOAD."
        ),
    )
    parser.add_argument(
        "--round-trip-cost-pct",
        type=float,
        default=None,
        help="Optional round-trip cost; half is used as per-side cost_pct.",
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        default=["baseline", "final"],
        help="Arms to report: baseline, final, all, or concrete arm names.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cost_pct = (
        float(args.round_trip_cost_pct) / 2.0
        if args.round_trip_cost_pct is not None
        else float(args.cost_pct)
    )
    os.environ["EPM_SIMPLE_POLICY_1M_DOWNLOAD"] = (
        "1" if bool(args.download_missing_1m) else "0"
    )
    os.environ["EPM_SIMPLE_POLICY_15M_DOWNLOAD"] = (
        "1" if bool(args.download_missing_1m) else "0"
    )
    raw_rows = _prepare_rows(args.candidates, min_rank=float(args.min_rank))
    state_column = _resolve_state_column(
        raw_rows,
        group_by=str(args.group_by),
        regime_column=str(args.regime_column),
        archetype_column=str(args.archetype_column),
    )
    bundles = _load_bundles(
        raw_rows,
        data_root=str(args.data_root),
        market_mode=str(args.market_mode),
        path_len=int(args.path_len),
        min_rows_per_strategy=int(args.min_rows_per_strategy),
    )
    replay_rows = pd.concat([bundle.rows for bundle in bundles], ignore_index=True)
    stage_rows = _load_stage_rows(args.walkforward_dir, args.arms)
    reports: Dict[str, pd.DataFrame] = {}
    aggregate_reports: Dict[str, pd.DataFrame] = {}
    for period in ("week", "month"):
        report = _build_period_report(
            raw_rows=raw_rows,
            replay_rows=replay_rows,
            stage_rows=stage_rows,
            bundles=bundles,
            period=period,
            group_by=str(args.group_by),
            archetype_column=state_column,
            regime_column=state_column,
            cost_pct=cost_pct,
            market_mode=str(args.market_mode),
            global_threshold_floor=float(args.global_threshold_floor),
        )
        reports[period] = report
        report.to_csv(args.out_dir / f"side_archetype_{period}_metrics.csv", index=False)
        aggregate = _aggregate_period_report(report)
        aggregate_reports[period] = aggregate
        aggregate.to_csv(
            args.out_dir / f"side_archetype_{period}_metrics_aggregated.csv",
            index=False,
        )

    manifest = {
        "generated_by": "report_side_archetype_policy_metrics",
        "candidate_path": str(args.candidates),
        "walkforward_dir": str(args.walkforward_dir),
        "out_dir": str(args.out_dir),
        "arms": list(args.arms),
        "raw_rows": int(len(raw_rows)),
        "replay_rows": int(len(replay_rows)),
        "path_survival_fraction": (
            float(len(replay_rows) / len(raw_rows)) if len(raw_rows) else 0.0
        ),
        "state_column": state_column,
        "cost_pct": float(cost_pct),
        "per_side_cost_pct": float(cost_pct),
        "round_trip_cost_pct": float(cost_pct * 2.0),
        "week_rows": int(len(reports["week"])),
        "month_rows": int(len(reports["month"])),
        "week_aggregated_rows": int(len(aggregate_reports["week"])),
        "month_aggregated_rows": int(len(aggregate_reports["month"])),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
