#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.ablate_simple_policy_exit_geometry import (
    DEFAULT_PATH_LEN,
    DEFAULT_POLICY_PER_SIDE_COST_PCT,
    _candidate_table_for_overrides,
    _load_bundles,
    _score_replay,
)


DEFAULT_SELECTED_ROWS = Path(
    "data_perp/reports/gmm_cluster_policy_smoke_20260702_wide_sidebalanced/"
    "gmm_train_meta_path_filter_smoke_s19_joint_meta_risk_monthly_gate_handoff/"
    "gmm_train_meta_path_filter_smoke_selected_rows.parquet"
)
DEFAULT_OUT_DIR = Path(
    "data_perp/reports/gmm_cluster_policy_smoke_20260702_wide_sidebalanced/"
    "meta_handoff_replay_attribution"
)
BAR_MINUTES = 15.0


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value) if not isinstance(value, (dict, list, tuple)) else False:
        return None
    return value


def _num(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _side_name(side: pd.Series) -> np.ndarray:
    values = pd.to_numeric(side, errors="coerce").fillna(1.0).to_numpy(dtype=np.float64)
    return np.where(values < 0.0, "short", "long")


def _side_code_from_frame(rows: pd.DataFrame) -> pd.Series:
    if "side" in rows.columns:
        return pd.to_numeric(rows["side"], errors="coerce").fillna(1.0)
    if "side_name" in rows.columns:
        text = rows["side_name"].astype(str).str.lower().str.strip()
        return pd.Series(np.where(text.eq("short"), -1.0, 1.0), index=rows.index)
    return pd.Series(1.0, index=rows.index)


def _first_numeric_column(rows: pd.DataFrame, columns: tuple[str, ...], default: float = np.nan) -> pd.Series:
    out = pd.Series(default, index=rows.index, dtype="float64")
    for column in columns:
        if column not in rows.columns:
            continue
        values = pd.to_numeric(rows[column], errors="coerce")
        out = out.where(out.notna(), values)
    return out


def _build_handoff(selected: pd.DataFrame, *, barrier_multiplier: float) -> pd.DataFrame:
    rows = selected.copy().reset_index(drop=True)
    rows["archetype_handoff_row_id"] = np.arange(len(rows), dtype=np.int64)
    timestamp = pd.to_datetime(rows["timestamp"], utc=True, errors="coerce")
    side = _side_code_from_frame(rows)
    meta_score = _first_numeric_column(
        rows,
        (
            "meta_selected_score",
            "meta_regime_score",
            "selected_score",
            "score",
        ),
    )
    score_rank = _first_numeric_column(
        rows,
        (
            "meta_score_rank_pct",
            "score_rank_pct_by_month",
            "score_rank_pct",
            "selector_rank_pct",
        ),
    )
    selected_rank = _first_numeric_column(
        rows,
        (
            "meta_score_rank_pct_selected",
            "selected_rank_pct",
            "score_rank_pct_by_month",
        ),
    )
    rank_pct = score_rank.fillna(selected_rank).fillna(1.0).clip(0.0, 1.0)
    barrier = _first_numeric_column(rows, ("barrier", "barrier_pct"), 0.005).fillna(0.005).clip(lower=1e-4)
    barrier = barrier * max(float(barrier_multiplier), 1e-6)
    strategy_id = (
        rows["strategy_id"].astype(str).to_numpy()
        if "strategy_id" in rows.columns
        else np.where(side.to_numpy(dtype=np.float64) < 0.0, "short", "long")
    )

    out = pd.DataFrame(
        {
            "timestamp": timestamp,
            "symbol": rows["symbol"].astype(str),
            "side": side.astype(np.float32),
            "strategy_id": strategy_id,
            "rank_pct": rank_pct.astype(np.float32),
            "calibrated_score": meta_score.astype(np.float32),
            "barrier_pct": barrier.astype(np.float32),
            "base_strategy_threshold": 0.0,
            "best_size_power": 1.0,
            "oof_regime_centroid_similarity_train": _first_numeric_column(
                rows,
                (
                    "joint_bad_risk",
                    "meta_regime_score",
                    "score_rank_pct_by_month",
                ),
                0.0,
            ).astype(np.float32),
            "archetype_handoff_row_id": rows["archetype_handoff_row_id"].astype(np.float32),
            "archetype_label_u_policy_net": _first_numeric_column(
                rows,
                ("u_policy_net", "ev_after_cost"),
            ).astype(np.float32),
            "archetype_label_ret_net": _first_numeric_column(
                rows,
                ("ret_net", "ev_after_cost"),
            ).astype(np.float32),
            "archetype_label_mae_norm": _first_numeric_column(
                rows,
                ("mae_norm", "bad_mae"),
            ).astype(np.float32),
            "archetype_label_mfe_norm": _first_numeric_column(
                rows,
                ("mfe_norm", "mfe_before_mae_1r"),
            ).astype(np.float32),
            "archetype_label_bad_mae_1r": _first_numeric_column(
                rows,
                ("bad_mae_1r", "bad_mae"),
                0.0,
            )
            .fillna(0.0)
            .astype(np.float32),
            "archetype_label_timeout": _first_numeric_column(
                rows,
                ("is_timeout", "timeout"),
                0.0,
            ),
            "archetype_meta_bad_risk": _first_numeric_column(
                rows,
                ("meta_bad_risk", "bad_mae_pred", "side_bad_mae_pred"),
            ).astype(np.float32),
            "archetype_meta_timeout_risk": _first_numeric_column(
                rows,
                ("meta_timeout_risk", "timeout_pred", "side_timeout_pred"),
            ).astype(np.float32),
            "archetype_joint_bad_risk": _first_numeric_column(
                rows,
                ("joint_bad_risk", "bad_mae_pred", "side_bad_mae_pred"),
            ).astype(np.float32),
            "archetype_joint_timeout_risk": _first_numeric_column(
                rows,
                ("joint_timeout_risk", "timeout_pred", "side_timeout_pred"),
            ).astype(np.float32),
        }
    )
    out["archetype_label_timeout"] = pd.to_numeric(
        out["archetype_label_timeout"],
        errors="coerce",
    ).fillna(0.0).astype(np.float32)
    out = out.dropna(subset=["timestamp", "symbol", "rank_pct", "barrier_pct"]).copy()
    out = out.sort_values(
        ["timestamp", "symbol", "strategy_id", "rank_pct", "calibrated_score"],
        ascending=[True, True, True, False, False],
        kind="mergesort",
    )
    out = out.drop_duplicates(["timestamp", "symbol", "strategy_id"], keep="first")
    return out.sort_values(["strategy_id", "timestamp", "symbol"]).reset_index(drop=True)


def _rate(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values)
    return float(np.nanmean(arr.astype(float))) if arr.size else float("nan")


def _mean(values: pd.Series | np.ndarray) -> float:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce")
    return float(numeric.mean()) if numeric.notna().any() else float("nan")


def _spearman(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    mask = x.notna() & y.notna()
    if int(mask.sum()) < 5:
        return float("nan")
    return float(x[mask].rank(method="average").corr(y[mask].rank(method="average")))


def _summarise_candidates(
    *,
    scenario: str,
    candidates: pd.DataFrame,
    decisions: pd.DataFrame,
    metrics: dict[str, Any],
    path_len: int,
    barrier_multiplier: float,
    delayed_entry_enabled: bool,
) -> dict[str, Any]:
    accepted = (
        decisions.loc[decisions["accepted"].astype(bool)].copy()
        if not decisions.empty and "accepted" in decisions.columns
        else pd.DataFrame()
    )
    exit_reason = candidates.get(
        "simple_policy_exit_reason",
        pd.Series("", index=candidates.index),
    ).astype(str)
    label_bad = _num(candidates, "archetype_label_bad_mae_1r", 0.0).fillna(0.0) > 0.5
    replay_full_sl = exit_reason.eq("full_sl")
    horizon_hours = float(path_len) * BAR_MINUTES / 60.0
    return {
        "scenario": scenario,
        "path_len": int(path_len),
        "horizon_hours": horizon_hours,
        "barrier_multiplier": float(barrier_multiplier),
        "delayed_entry_enabled": bool(delayed_entry_enabled),
        "candidate_rows": int(len(candidates)),
        "accepted_trades": int(len(accepted)),
        "label_mean_u": _mean(candidates.get("archetype_label_u_policy_net", [])),
        "label_bad_mae_1r_rate": _rate(label_bad),
        "label_timeout_rate": _rate(
            _num(candidates, "archetype_label_timeout", 0.0).fillna(0.0) > 0.5
        ),
        "replay_candidate_mean_net_return": _mean(candidates.get("net_return", [])),
        "replay_candidate_gross_mean_return": _mean(candidates.get("gross_return", [])),
        "replay_full_sl_rate": _rate(replay_full_sl),
        "replay_timeout_rate": _rate(exit_reason.eq("timeout")),
        "replay_trailing_rate": _rate(exit_reason.eq("trailing")),
        "replay_capital_protect_rate": _rate(exit_reason.eq("capital_protect")),
        "replay_hard_tp_rate": _rate(exit_reason.eq("hard_tp")),
        "label_bad_and_replay_full_sl_rate": _rate(label_bad & replay_full_sl),
        "label_clean_but_replay_full_sl_rate": _rate((~label_bad) & replay_full_sl),
        "label_replay_net_spearman": _spearman(
            candidates.get("archetype_label_u_policy_net", pd.Series(dtype=float)),
            candidates.get("net_return", pd.Series(dtype=float)),
        ),
        "barrier_mean": _mean(candidates.get("barrier_pct", [])),
        "policy_sl_return_mean": _mean(candidates.get("policy_sl_return", [])),
        "entry_reanchor_bps_mean": _mean(candidates.get("entry_reanchor_bps", [])),
        "expected_friction_bps_mean": _mean(candidates.get("expected_friction_bps", [])),
        "portfolio_objective": float(metrics.get("objective", np.nan)),
        "portfolio_net_pnl": float(metrics.get("net_pnl", np.nan)),
        "portfolio_compounded_return": float(metrics.get("compounded_return", np.nan)),
        "portfolio_full_sl_rate": float(metrics.get("full_sl_rate", np.nan)),
        "portfolio_timeout_rate": float(metrics.get("timeout_rate", np.nan)),
        "portfolio_max_drawdown": float(metrics.get("max_drawdown", np.nan)),
    }


def _group_summary(frame: pd.DataFrame, group_cols: list[str], scenario: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    work = frame.copy()
    work["month"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce").dt.to_period("M").astype(str)
    work["label_bad"] = _num(work, "archetype_label_bad_mae_1r", 0.0).fillna(0.0) > 0.5
    work["label_timeout"] = _num(work, "archetype_label_timeout", 0.0).fillna(0.0) > 0.5
    work["replay_full_sl"] = work["simple_policy_exit_reason"].astype(str).eq("full_sl")
    work["replay_timeout"] = work["simple_policy_exit_reason"].astype(str).eq("timeout")
    rows: list[dict[str, Any]] = []
    for key, group in work.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        row = {"scenario": scenario, **{col: val for col, val in zip(group_cols, key)}}
        row.update(
            {
                "rows": int(len(group)),
                "label_mean_u": _mean(group.get("archetype_label_u_policy_net", pd.Series(dtype=float))),
                "label_bad_mae_1r_rate": _rate(group["label_bad"]),
                "label_timeout_rate": _rate(group["label_timeout"]),
                "replay_mean_net_return": _mean(group["net_return"]),
                "replay_full_sl_rate": _rate(group["replay_full_sl"]),
                "replay_timeout_rate": _rate(group["replay_timeout"]),
                "label_replay_net_spearman": _spearman(
                    group.get("archetype_label_u_policy_net", pd.Series(dtype=float)),
                    group["net_return"],
                ),
                "barrier_mean": _mean(group["barrier_pct"]),
                "policy_sl_return_mean": _mean(group["policy_sl_return"]),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _oracle_summary(
    candidates: pd.DataFrame,
    *,
    top_ks: tuple[int, ...] = (1, 3, 5),
) -> pd.DataFrame:
    if candidates.empty or "scenario" not in candidates.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    work = candidates.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work["net_return"] = pd.to_numeric(work.get("net_return"), errors="coerce")
    work["label_u"] = pd.to_numeric(
        work.get("archetype_label_u_policy_net"),
        errors="coerce",
    )
    for scenario, scenario_frame in work.groupby("scenario", dropna=False, sort=True):
        scenario_frame = scenario_frame.dropna(subset=["timestamp", "net_return"])
        if scenario_frame.empty:
            continue
        base = {
            "scenario": str(scenario),
            "path_len": int(_mean(scenario_frame.get("path_len", []))),
            "horizon_hours": _mean(scenario_frame.get("horizon_hours", [])),
            "barrier_multiplier": _mean(scenario_frame.get("barrier_multiplier", [])),
            "candidate_rows": int(len(scenario_frame)),
            "timestamps": int(scenario_frame["timestamp"].nunique()),
            "mean_candidates_per_timestamp": float(
                len(scenario_frame) / max(int(scenario_frame["timestamp"].nunique()), 1)
            ),
            "candidate_mean_net_return": _mean(scenario_frame["net_return"]),
            "candidate_hit_net_rate": _rate(scenario_frame["net_return"] > 0.0),
            "rank_pct_replay_net_spearman": _spearman(
                scenario_frame.get("rank_pct", pd.Series(dtype=float)),
                scenario_frame["net_return"],
            ),
            "label_replay_net_spearman": _spearman(
                scenario_frame["label_u"],
                scenario_frame["net_return"],
            ),
        }
        for top_k in top_ks:
            top_net: list[float] = []
            top_label: list[float] = []
            for _timestamp, group in scenario_frame.groupby("timestamp", sort=False):
                selected = group.sort_values("net_return", ascending=False).head(int(top_k))
                top_net.append(_mean(selected["net_return"]))
                top_label.append(_mean(selected["label_u"]))
            top_net_ser = pd.Series(top_net, dtype="float64")
            top_label_ser = pd.Series(top_label, dtype="float64")
            row = dict(base)
            row.update(
                {
                    "oracle_top_k": int(top_k),
                    "oracle_mean_net_return": _mean(top_net_ser),
                    "oracle_hit_net_rate": _rate(top_net_ser > 0.0),
                    "oracle_q10_net_return": float(top_net_ser.quantile(0.10))
                    if top_net_ser.notna().any()
                    else float("nan"),
                    "oracle_label_u_mean": _mean(top_label_ser),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _run_scenario(
    *,
    selected_rows: pd.DataFrame,
    barrier_multiplier: float,
    delayed_entry_enabled: bool,
    data_root: str,
    market_mode: str,
    path_len: int,
    min_rows_per_strategy: int,
    cost_pct: float,
    global_threshold_floor: float,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    os.environ["EPM_SIMPLE_POLICY_1M_EXECUTION_ENABLED"] = (
        "1" if delayed_entry_enabled else "0"
    )
    rows = _build_handoff(selected_rows, barrier_multiplier=barrier_multiplier)
    bundles = _load_bundles(
        rows,
        data_root=data_root,
        market_mode=market_mode,
        path_len=path_len,
        min_rows_per_strategy=min_rows_per_strategy,
    )
    frames: list[pd.DataFrame] = []
    for bundle in bundles:
        frame = _candidate_table_for_overrides(
            [bundle],
            overrides={},
            cost_pct=cost_pct,
            market_mode=market_mode,
            arm="baseline",
        )
        if not frame.empty:
            frames.append(frame)
    candidates = (
        pd.concat(frames, ignore_index=True).sort_values(["timestamp", "strategy_id", "symbol"])
        if frames
        else pd.DataFrame()
    )
    decisions, _equity, metrics = _score_replay(
        candidates,
        market_mode=market_mode,
        global_threshold_floor=global_threshold_floor,
    )
    scenario = f"delay_{int(delayed_entry_enabled)}_barrier_x{barrier_multiplier:g}"
    if not candidates.empty:
        horizon_hours = float(path_len) * BAR_MINUTES / 60.0
        scenario = (
            f"h{horizon_hours:g}_delay_{int(delayed_entry_enabled)}"
            f"_barrier_x{barrier_multiplier:g}"
        )
        candidates["scenario"] = scenario
        candidates["path_len"] = int(path_len)
        candidates["horizon_hours"] = horizon_hours
        candidates["barrier_multiplier"] = float(barrier_multiplier)
        candidates["delayed_entry_enabled"] = bool(delayed_entry_enabled)
    else:
        horizon_hours = float(path_len) * BAR_MINUTES / 60.0
        scenario = (
            f"h{horizon_hours:g}_delay_{int(delayed_entry_enabled)}"
            f"_barrier_x{barrier_multiplier:g}"
        )
    summary = _summarise_candidates(
        scenario=scenario,
        candidates=candidates,
        decisions=decisions,
        metrics=metrics,
        path_len=int(path_len),
        barrier_multiplier=barrier_multiplier,
        delayed_entry_enabled=delayed_entry_enabled,
    )
    by_month_side = _group_summary(candidates, ["month", "side"], scenario)
    by_side = _group_summary(candidates, ["side"], scenario)
    return summary, by_month_side, by_side, candidates


def _parse_path_lens(value: str | None, fallback: int) -> list[int]:
    if value is None or not str(value).strip():
        return [int(fallback)]
    out: list[int] = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        parsed = int(part)
        if parsed <= 0:
            raise ValueError(f"path_lens must be positive; got {parsed}")
        out.append(parsed)
    if not out:
        return [int(fallback)]
    return sorted(set(out))


def _apply_selected_filters(
    selected: pd.DataFrame,
    *,
    max_meta_rank_pct: float | None,
    max_selected_rank_pct: float | None,
    max_joint_bad_risk: float | None,
    max_joint_timeout_risk: float | None,
    max_meta_bad_risk: float | None,
    max_meta_timeout_risk: float | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    mask = pd.Series(True, index=selected.index)
    filters: dict[str, float] = {}
    for column, value, key in (
        ("meta_score_rank_pct", max_meta_rank_pct, "max_meta_rank_pct"),
        ("meta_score_rank_pct_selected", max_selected_rank_pct, "max_selected_rank_pct"),
        ("joint_bad_risk", max_joint_bad_risk, "max_joint_bad_risk"),
        ("joint_timeout_risk", max_joint_timeout_risk, "max_joint_timeout_risk"),
        ("meta_bad_risk", max_meta_bad_risk, "max_meta_bad_risk"),
        ("meta_timeout_risk", max_meta_timeout_risk, "max_meta_timeout_risk"),
    ):
        if value is None:
            continue
        if column not in selected.columns:
            raise ValueError(f"Cannot filter selected rows: missing column {column!r}")
        threshold = float(value)
        filters[key] = threshold
        mask &= pd.to_numeric(selected[column], errors="coerce").le(threshold)
    out = selected.loc[mask].reset_index(drop=True).copy()
    return out, {
        "input_rows": int(len(selected)),
        "output_rows": int(len(out)),
        "retention_rate": float(len(out) / max(len(selected), 1)),
        "filters": filters,
    }


def _optional_float(value: float) -> float | None:
    return None if value < 0.0 else float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-rows", type=Path, default=DEFAULT_SELECTED_ROWS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--data-root", default="data_perp")
    parser.add_argument("--market-mode", default="perps", choices=["spot", "perps"])
    parser.add_argument("--exchange", default="krakenfutures")
    parser.add_argument("--path-len", type=int, default=DEFAULT_PATH_LEN)
    parser.add_argument(
        "--path-lens",
        default=None,
        help=(
            "Optional comma-separated path lengths to sweep. Each bar is 15m in "
            "the simple_policy replay store, so 12,16,20,24,28 cover 3-7h."
        ),
    )
    parser.add_argument("--min-rows-per-strategy", type=int, default=5)
    parser.add_argument("--cost-pct", type=float, default=DEFAULT_POLICY_PER_SIDE_COST_PCT)
    parser.add_argument("--global-threshold-floor", type=float, default=0.0)
    parser.add_argument("--barrier-multipliers", default="1,1.5,2,3,4")
    parser.add_argument("--include-no-delay", action="store_true")
    parser.add_argument(
        "--download-missing-1m",
        action="store_true",
        help="Allow simple_policy replay to download missing 1m candles for path construction.",
    )
    parser.add_argument(
        "--max-meta-rank-pct",
        type=float,
        default=-1.0,
        help="Optional selected-row filter on meta_score_rank_pct; negative disables.",
    )
    parser.add_argument(
        "--max-selected-rank-pct",
        type=float,
        default=-1.0,
        help="Optional selected-row filter on meta_score_rank_pct_selected; negative disables.",
    )
    parser.add_argument(
        "--max-joint-bad-risk",
        type=float,
        default=-1.0,
        help="Optional selected-row filter on joint_bad_risk; negative disables.",
    )
    parser.add_argument(
        "--max-joint-timeout-risk",
        type=float,
        default=-1.0,
        help="Optional selected-row filter on joint_timeout_risk; negative disables.",
    )
    parser.add_argument(
        "--max-meta-bad-risk",
        type=float,
        default=-1.0,
        help="Optional selected-row filter on meta_bad_risk; negative disables.",
    )
    parser.add_argument(
        "--max-meta-timeout-risk",
        type=float,
        default=-1.0,
        help="Optional selected-row filter on meta_timeout_risk; negative disables.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for key in ("EPM_EXCHANGE", "EXCHANGE_NAME", "PRIMARY_EXCHANGE"):
        os.environ[key] = str(args.exchange)
    os.environ["EPM_SIMPLE_POLICY_1M_DOWNLOAD"] = "1" if bool(args.download_missing_1m) else "0"

    selected_raw = pd.read_parquet(args.selected_rows)
    selected, selected_filter_summary = _apply_selected_filters(
        selected_raw,
        max_meta_rank_pct=_optional_float(float(args.max_meta_rank_pct)),
        max_selected_rank_pct=_optional_float(float(args.max_selected_rank_pct)),
        max_joint_bad_risk=_optional_float(float(args.max_joint_bad_risk)),
        max_joint_timeout_risk=_optional_float(float(args.max_joint_timeout_risk)),
        max_meta_bad_risk=_optional_float(float(args.max_meta_bad_risk)),
        max_meta_timeout_risk=_optional_float(float(args.max_meta_timeout_risk)),
    )
    if selected.empty:
        raise ValueError(f"No selected rows left after filters: {selected_filter_summary}")
    path_lens = _parse_path_lens(args.path_lens, int(args.path_len))
    multipliers = [
        float(part)
        for part in str(args.barrier_multipliers).split(",")
        if str(part).strip()
    ]
    delayed_options = [True, False] if bool(args.include_no_delay) else [True]

    summaries: list[dict[str, Any]] = []
    monthly_frames: list[pd.DataFrame] = []
    side_frames: list[pd.DataFrame] = []
    candidate_frames: list[pd.DataFrame] = []
    for path_len in path_lens:
        for delayed_enabled in delayed_options:
            for multiplier in multipliers:
                summary, by_month_side, by_side, candidates = _run_scenario(
                    selected_rows=selected,
                    barrier_multiplier=multiplier,
                    delayed_entry_enabled=delayed_enabled,
                    data_root=str(args.data_root),
                    market_mode=str(args.market_mode),
                    path_len=int(path_len),
                    min_rows_per_strategy=int(args.min_rows_per_strategy),
                    cost_pct=float(args.cost_pct),
                    global_threshold_floor=float(args.global_threshold_floor),
                )
                summaries.append(summary)
                monthly_frames.append(by_month_side)
                side_frames.append(by_side)
                candidate_frames.append(candidates)

    summary_df = pd.DataFrame(summaries)
    monthly_df = pd.concat(monthly_frames, ignore_index=True) if monthly_frames else pd.DataFrame()
    side_df = pd.concat(side_frames, ignore_index=True) if side_frames else pd.DataFrame()
    candidates_df = (
        pd.concat(candidate_frames, ignore_index=True)
        if candidate_frames
        else pd.DataFrame()
    )
    paths = {
        "scenario_summary": args.out_dir / "meta_handoff_replay_attribution_summary.csv",
        "month_side": args.out_dir / "meta_handoff_replay_attribution_month_side.csv",
        "side": args.out_dir / "meta_handoff_replay_attribution_side.csv",
        "oracle_summary": args.out_dir / "meta_handoff_replay_oracle_summary.csv",
        "candidate_rows": args.out_dir / "meta_handoff_replay_attribution_candidates.parquet",
        "manifest": args.out_dir / "manifest.json",
    }
    oracle_df = _oracle_summary(candidates_df)
    summary_df.to_csv(paths["scenario_summary"], index=False)
    monthly_df.to_csv(paths["month_side"], index=False)
    side_df.to_csv(paths["side"], index=False)
    oracle_df.to_csv(paths["oracle_summary"], index=False)
    candidates_df.to_parquet(paths["candidate_rows"], index=False)
    best = (
        summary_df.sort_values(
            ["portfolio_net_pnl", "replay_candidate_mean_net_return"],
            ascending=[False, False],
        )
        .head(1)
        .to_dict(orient="records")
    )
    manifest = {
        "generated_by": "report_meta_handoff_replay_attribution",
        "selected_rows": str(args.selected_rows),
        "out_dir": str(args.out_dir),
        "exchange": str(args.exchange),
        "market_mode": str(args.market_mode),
        "path_lens": path_lens,
        "horizon_hours": [float(v) * BAR_MINUTES / 60.0 for v in path_lens],
        "selected_row_filter": selected_filter_summary,
        "barrier_multipliers": multipliers,
        "include_no_delay": bool(args.include_no_delay),
        "download_missing_1m": bool(args.download_missing_1m),
        "best_scenario": best[0] if best else None,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    print(json.dumps(_json_safe({"best_scenario": manifest["best_scenario"], "outputs": manifest["outputs"]}), indent=2))


if __name__ == "__main__":
    main()
