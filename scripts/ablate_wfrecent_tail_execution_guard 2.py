#!/usr/bin/env python3
"""Walk-forward tail/execution guard around the fixed wf_recent TP/SL combo.

The guard is intentionally narrow: default to ``wf_recent`` and fall back to
``static`` only when prior similar diagnostic states suggest that ``wf_recent``
will hurt full-SL/timeout behavior without enough net-PnL compensation.

This script reuses existing materialized weekly comparison artifacts. It does
not rerun the portfolio replay.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "recent_hr_surprise": (
        "prev_diag_recent_hr_surprise_",
        "roll2_diag_recent_hr_surprise_",
        "roll3_diag_recent_hr_surprise_",
    ),
    "uncertainty": (
        "prev_diag_uncertainty_",
        "roll2_diag_uncertainty_",
        "roll3_diag_uncertainty_",
    ),
    "ood": (
        "prev_diag_ood_",
        "roll2_diag_ood_",
        "roll3_diag_ood_",
    ),
    "drift": (
        "prev_diag_drift_",
        "roll2_diag_drift_",
        "roll3_diag_drift_",
    ),
    "recent_hr_uncertainty": (
        "prev_diag_recent_hr_surprise_",
        "roll2_diag_recent_hr_surprise_",
        "roll3_diag_recent_hr_surprise_",
        "prev_diag_uncertainty_",
        "roll2_diag_uncertainty_",
        "roll3_diag_uncertainty_",
    ),
    "all_diagnostics": ("prev_diag_", "roll2_diag_", "roll3_diag_"),
}


@dataclass(frozen=True)
class GuardConfig:
    feature_set: str
    k_neighbors: int
    min_neighbor_count: int
    min_train_weeks: int
    full_sl_gate: float
    timeout_gate: float
    min_net_delta: float
    min_q20_delta: float
    min_positive_share: float
    mode: str


def _objective(values: np.ndarray, q35_weight: float, q20_weight: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.mean(values) + q35_weight * np.quantile(values, 0.35) + q20_weight * np.quantile(values, 0.20))


def _fmt_table(frame: pd.DataFrame, cols: list[str], max_rows: int | None = None) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame[cols].head(max_rows).copy() if max_rows else frame[cols].copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:,.3f}")
    return view.to_markdown(index=False)


def _read_weekly(path: Path) -> pd.DataFrame:
    weekly = pd.read_csv(path)
    weekly = weekly[weekly["label"].isin(["static", "wf_recent"])].copy()
    weekly["week_start"] = weekly["week"].astype(str).str.split("/").str[0]
    weekly["week_start"] = pd.to_datetime(weekly["week_start"], errors="coerce")
    for col in (
        "net_pnl",
        "gross_pnl",
        "trades",
        "hit_rate",
        "full_sl_rate",
        "timeout_rate",
        "delta_net_pnl",
        "delta_full_sl_rate",
        "delta_timeout_rate",
    ):
        if col in weekly.columns:
            weekly[col] = pd.to_numeric(weekly[col], errors="coerce")
    return weekly.sort_values(["week_start", "label"]).reset_index(drop=True)


def _load_signals(path: Path) -> pd.DataFrame:
    signals = pd.read_csv(path)
    signals["week_start"] = pd.to_datetime(signals["week_start"], errors="coerce")
    return signals[signals["week_start"].notna()].sort_values("week_start").reset_index(drop=True)


def _wide_weekly(weekly: pd.DataFrame) -> pd.DataFrame:
    metrics = ["net_pnl", "gross_pnl", "trades", "hit_rate", "full_sl_rate", "timeout_rate"]
    wide = weekly.pivot(index="week", columns="label", values=metrics)
    wide.columns = [f"{metric}__{label}" for metric, label in wide.columns]
    week_starts = weekly.drop_duplicates("week")[["week", "week_start"]].set_index("week")
    wide = week_starts.join(wide, how="inner").reset_index()
    wide["delta_net_pnl__wf_minus_static"] = wide["net_pnl__wf_recent"] - wide["net_pnl__static"]
    wide["delta_full_sl_rate__wf_minus_static"] = wide["full_sl_rate__wf_recent"] - wide["full_sl_rate__static"]
    wide["delta_timeout_rate__wf_minus_static"] = wide["timeout_rate__wf_recent"] - wide["timeout_rate__static"]
    wide["delta_hit_rate__wf_minus_static"] = wide["hit_rate__wf_recent"] - wide["hit_rate__static"]
    wide["delta_trades__wf_minus_static"] = wide["trades__wf_recent"] - wide["trades__static"]
    return wide.sort_values("week_start").reset_index(drop=True)


def _feature_columns(signals: pd.DataFrame, feature_set: str) -> list[str]:
    prefixes = FEATURE_SETS[feature_set]
    return [
        col
        for col in signals.columns
        if col not in {"week", "week_start"} and any(col.startswith(prefix) for prefix in prefixes)
    ]


def _distances(train_x: pd.DataFrame, current_x: pd.Series) -> np.ndarray:
    arr = train_x.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    cur = pd.to_numeric(current_x, errors="coerce").to_numpy(dtype=np.float64)
    if arr.size == 0:
        return np.array([], dtype=np.float64)
    finite_col = np.isfinite(arr).sum(axis=0) >= max(4, int(0.5 * arr.shape[0]))
    if not finite_col.any():
        return np.full(arr.shape[0], np.inf, dtype=np.float64)
    arr = arr[:, finite_col]
    cur = cur[finite_col]
    med = np.nanmedian(arr, axis=0)
    q25 = np.nanquantile(arr, 0.25, axis=0)
    q75 = np.nanquantile(arr, 0.75, axis=0)
    scale = q75 - q25
    fallback = np.nanstd(arr, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-9), scale, fallback)
    good = np.isfinite(scale) & (scale > 1e-9) & np.isfinite(cur)
    if not good.any():
        return np.full(arr.shape[0], np.inf, dtype=np.float64)
    diff = (arr[:, good] - med[good]) / scale[good] - ((cur[good] - med[good]) / scale[good])[None, :]
    valid = np.isfinite(diff)
    counts = valid.sum(axis=1)
    sq = np.where(valid, diff * diff, 0.0).sum(axis=1)
    out = np.full(arr.shape[0], np.inf, dtype=np.float64)
    np.divide(sq, counts, out=out, where=counts > 0)
    return out


def _configs() -> list[GuardConfig]:
    configs: list[GuardConfig] = []
    for feature_set in FEATURE_SETS:
        for k_neighbors in (3, 5):
            for min_train_weeks in (8, 12):
                for full_sl_gate in (0.01, 0.02):
                    for timeout_gate in (0.04, 0.08):
                        for min_net_delta in (-500.0, 0.0, 500.0):
                            for min_q20_delta in (-250.0, 0.0):
                                for min_positive_share in (0.50, 0.60):
                                    for mode in ("full_sl_or_timeout", "execution_and_net_weak"):
                                        configs.append(
                                            GuardConfig(
                                                feature_set=feature_set,
                                                k_neighbors=k_neighbors,
                                                min_neighbor_count=max(3, min(k_neighbors, 5)),
                                                min_train_weeks=min_train_weeks,
                                                full_sl_gate=full_sl_gate,
                                                timeout_gate=timeout_gate,
                                                min_net_delta=min_net_delta,
                                                min_q20_delta=min_q20_delta,
                                                min_positive_share=min_positive_share,
                                                mode=mode,
                                            )
                                        )
    return configs


def _guard_triggers(stats: dict[str, float], cfg: GuardConfig) -> bool:
    full_sl_bad = stats["mean_delta_full_sl_rate"] >= cfg.full_sl_gate
    timeout_bad = stats["mean_delta_timeout_rate"] >= cfg.timeout_gate
    net_weak = stats["mean_delta_net_pnl"] <= cfg.min_net_delta
    q20_weak = stats["q20_delta_net_pnl"] <= cfg.min_q20_delta
    positive_weak = stats["positive_share_net_pnl"] <= cfg.min_positive_share
    if cfg.mode == "full_sl_or_timeout":
        return (full_sl_bad or timeout_bad) and (net_weak or q20_weak or positive_weak)
    if cfg.mode == "full_sl_and_timeout":
        return full_sl_bad and timeout_bad and (net_weak or q20_weak or positive_weak)
    if cfg.mode == "execution_and_net_weak":
        return (full_sl_bad or timeout_bad) and net_weak and (q20_weak or positive_weak)
    raise ValueError(f"Unknown mode {cfg.mode}")


def _evaluate(
    weekly_wide: pd.DataFrame,
    signals: pd.DataFrame,
    cfg: GuardConfig,
    q35_weight: float,
    q20_weight: float,
    start_week: str | None = None,
    end_week: str | None = None,
) -> pd.DataFrame:
    signals = signals.merge(weekly_wide[["week", "week_start"]], on=["week", "week_start"], how="inner")
    feature_cols = _feature_columns(signals, cfg.feature_set)
    if not feature_cols:
        raise ValueError(f"No diagnostic features found for {cfg.feature_set}")
    rows: list[dict[str, object]] = []
    start_ts = pd.Timestamp(start_week) if start_week else None
    end_ts = pd.Timestamp(end_week) if end_week else None
    for pos, sig_row in signals.iterrows():
        week_start = pd.Timestamp(sig_row["week_start"])
        if pos < cfg.min_train_weeks:
            continue
        if start_ts is not None and week_start < start_ts:
            continue
        if end_ts is not None and week_start >= end_ts:
            continue
        train_signals = signals.iloc[:pos].reset_index(drop=True)
        train_weekly = weekly_wide[weekly_wide["week"].isin(train_signals["week"])].reset_index(drop=True)
        distances = _distances(train_signals[feature_cols], sig_row[feature_cols])
        order = np.argsort(distances)
        order = order[np.isfinite(distances[order])][: cfg.k_neighbors]
        selected_label = "wf_recent"
        trigger = False
        stats = {
            "neighbor_count": int(len(order)),
            "mean_delta_net_pnl": 0.0,
            "q20_delta_net_pnl": 0.0,
            "positive_share_net_pnl": 0.0,
            "mean_delta_full_sl_rate": 0.0,
            "mean_delta_timeout_rate": 0.0,
            "mean_delta_hit_rate": 0.0,
        }
        if len(order) >= cfg.min_neighbor_count:
            neighbor = train_weekly.iloc[order].copy()
            dnet = neighbor["delta_net_pnl__wf_minus_static"].to_numpy(dtype=np.float64)
            dsl = neighbor["delta_full_sl_rate__wf_minus_static"].to_numpy(dtype=np.float64)
            dto = neighbor["delta_timeout_rate__wf_minus_static"].to_numpy(dtype=np.float64)
            dhr = neighbor["delta_hit_rate__wf_minus_static"].to_numpy(dtype=np.float64)
            finite_net = dnet[np.isfinite(dnet)]
            stats = {
                "neighbor_count": int(len(neighbor)),
                "mean_delta_net_pnl": float(np.nanmean(dnet)),
                "q20_delta_net_pnl": float(np.nanquantile(dnet, 0.20)),
                "positive_share_net_pnl": float(np.mean(finite_net > 0.0)) if finite_net.size else 0.0,
                "mean_delta_full_sl_rate": float(np.nanmean(dsl)),
                "mean_delta_timeout_rate": float(np.nanmean(dto)),
                "mean_delta_hit_rate": float(np.nanmean(dhr)),
            }
            trigger = _guard_triggers(stats, cfg)
            if trigger:
                selected_label = "static"
        current = weekly_wide[weekly_wide["week"].eq(sig_row["week"])].iloc[0]
        selected_net = float(current[f"net_pnl__{selected_label}"])
        wf_net = float(current["net_pnl__wf_recent"])
        static_net = float(current["net_pnl__static"])
        rows.append(
            {
                "week": sig_row["week"],
                "week_start": week_start.date().isoformat(),
                "selected_label": selected_label,
                "fallback_to_static": bool(trigger),
                "net_pnl": selected_net,
                "wf_recent_net_pnl": wf_net,
                "static_net_pnl": static_net,
                "delta_vs_wf_recent_net_pnl": selected_net - wf_net,
                "delta_vs_static_net_pnl": selected_net - static_net,
                "trades": float(current[f"trades__{selected_label}"]),
                "hit_rate": float(current[f"hit_rate__{selected_label}"]),
                "full_sl_rate": float(current[f"full_sl_rate__{selected_label}"]),
                "timeout_rate": float(current[f"timeout_rate__{selected_label}"]),
                "wf_recent_full_sl_rate": float(current["full_sl_rate__wf_recent"]),
                "static_full_sl_rate": float(current["full_sl_rate__static"]),
                "wf_recent_timeout_rate": float(current["timeout_rate__wf_recent"]),
                "static_timeout_rate": float(current["timeout_rate__static"]),
                "wf_recent_hit_rate": float(current["hit_rate__wf_recent"]),
                "static_hit_rate": float(current["hit_rate__static"]),
                **stats,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["objective_component"] = out["net_pnl"]
        out.attrs["objective"] = _objective(out["net_pnl"].to_numpy(dtype=np.float64), q35_weight, q20_weight)
    return out


def _precompute_neighbor_base(
    weekly_wide: pd.DataFrame,
    signals: pd.DataFrame,
    feature_set: str,
    k_neighbors: int,
    min_train_weeks: int,
    start_week: str | None = None,
    end_week: str | None = None,
) -> pd.DataFrame:
    signals = signals.merge(weekly_wide[["week", "week_start"]], on=["week", "week_start"], how="inner")
    feature_cols = _feature_columns(signals, feature_set)
    if not feature_cols:
        raise ValueError(f"No diagnostic features found for {feature_set}")
    start_ts = pd.Timestamp(start_week) if start_week else None
    end_ts = pd.Timestamp(end_week) if end_week else None
    rows: list[dict[str, object]] = []
    for pos, sig_row in signals.iterrows():
        week_start = pd.Timestamp(sig_row["week_start"])
        if pos < min_train_weeks:
            continue
        if start_ts is not None and week_start < start_ts:
            continue
        if end_ts is not None and week_start >= end_ts:
            continue
        train_signals = signals.iloc[:pos].reset_index(drop=True)
        train_weekly = weekly_wide[weekly_wide["week"].isin(train_signals["week"])].reset_index(drop=True)
        distances = _distances(train_signals[feature_cols], sig_row[feature_cols])
        order = np.argsort(distances)
        order = order[np.isfinite(distances[order])][:k_neighbors]
        stats = {
            "neighbor_count": int(len(order)),
            "mean_delta_net_pnl": 0.0,
            "q20_delta_net_pnl": 0.0,
            "positive_share_net_pnl": 0.0,
            "mean_delta_full_sl_rate": 0.0,
            "mean_delta_timeout_rate": 0.0,
            "mean_delta_hit_rate": 0.0,
        }
        if len(order) > 0:
            neighbor = train_weekly.iloc[order].copy()
            dnet = neighbor["delta_net_pnl__wf_minus_static"].to_numpy(dtype=np.float64)
            dsl = neighbor["delta_full_sl_rate__wf_minus_static"].to_numpy(dtype=np.float64)
            dto = neighbor["delta_timeout_rate__wf_minus_static"].to_numpy(dtype=np.float64)
            dhr = neighbor["delta_hit_rate__wf_minus_static"].to_numpy(dtype=np.float64)
            finite_net = dnet[np.isfinite(dnet)]
            stats = {
                "neighbor_count": int(len(neighbor)),
                "mean_delta_net_pnl": float(np.nanmean(dnet)),
                "q20_delta_net_pnl": float(np.nanquantile(dnet, 0.20)),
                "positive_share_net_pnl": float(np.mean(finite_net > 0.0)) if finite_net.size else 0.0,
                "mean_delta_full_sl_rate": float(np.nanmean(dsl)),
                "mean_delta_timeout_rate": float(np.nanmean(dto)),
                "mean_delta_hit_rate": float(np.nanmean(dhr)),
            }
        current = weekly_wide[weekly_wide["week"].eq(sig_row["week"])].iloc[0]
        rows.append(
            {
                "week": sig_row["week"],
                "week_start": week_start.date().isoformat(),
                "net_pnl__wf_recent": float(current["net_pnl__wf_recent"]),
                "net_pnl__static": float(current["net_pnl__static"]),
                "trades__wf_recent": float(current["trades__wf_recent"]),
                "trades__static": float(current["trades__static"]),
                "hit_rate__wf_recent": float(current["hit_rate__wf_recent"]),
                "hit_rate__static": float(current["hit_rate__static"]),
                "full_sl_rate__wf_recent": float(current["full_sl_rate__wf_recent"]),
                "full_sl_rate__static": float(current["full_sl_rate__static"]),
                "timeout_rate__wf_recent": float(current["timeout_rate__wf_recent"]),
                "timeout_rate__static": float(current["timeout_rate__static"]),
                **stats,
            }
        )
    return pd.DataFrame(rows)


def _evaluate_cached(base: pd.DataFrame, cfg: GuardConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, base_row in base.iterrows():
        stats = {
            "neighbor_count": int(base_row["neighbor_count"]),
            "mean_delta_net_pnl": float(base_row["mean_delta_net_pnl"]),
            "q20_delta_net_pnl": float(base_row["q20_delta_net_pnl"]),
            "positive_share_net_pnl": float(base_row["positive_share_net_pnl"]),
            "mean_delta_full_sl_rate": float(base_row["mean_delta_full_sl_rate"]),
            "mean_delta_timeout_rate": float(base_row["mean_delta_timeout_rate"]),
            "mean_delta_hit_rate": float(base_row["mean_delta_hit_rate"]),
        }
        trigger = stats["neighbor_count"] >= cfg.min_neighbor_count and _guard_triggers(stats, cfg)
        selected_label = "static" if trigger else "wf_recent"
        wf_net = float(base_row["net_pnl__wf_recent"])
        static_net = float(base_row["net_pnl__static"])
        selected_net = static_net if trigger else wf_net
        rows.append(
            {
                "week": base_row["week"],
                "week_start": base_row["week_start"],
                "selected_label": selected_label,
                "fallback_to_static": bool(trigger),
                "net_pnl": selected_net,
                "wf_recent_net_pnl": wf_net,
                "static_net_pnl": static_net,
                "delta_vs_wf_recent_net_pnl": selected_net - wf_net,
                "delta_vs_static_net_pnl": selected_net - static_net,
                "trades": float(base_row[f"trades__{selected_label}"]),
                "hit_rate": float(base_row[f"hit_rate__{selected_label}"]),
                "full_sl_rate": float(base_row[f"full_sl_rate__{selected_label}"]),
                "timeout_rate": float(base_row[f"timeout_rate__{selected_label}"]),
                "wf_recent_full_sl_rate": float(base_row["full_sl_rate__wf_recent"]),
                "static_full_sl_rate": float(base_row["full_sl_rate__static"]),
                "wf_recent_timeout_rate": float(base_row["timeout_rate__wf_recent"]),
                "static_timeout_rate": float(base_row["timeout_rate__static"]),
                "wf_recent_hit_rate": float(base_row["hit_rate__wf_recent"]),
                "static_hit_rate": float(base_row["hit_rate__static"]),
                **stats,
            }
        )
    return pd.DataFrame(rows)


def _summarize(rows: pd.DataFrame, cfg: GuardConfig, run_id: int, q35_weight: float, q20_weight: float) -> dict[str, object]:
    values = rows["net_pnl"].to_numpy(dtype=np.float64)
    wf_values = rows["wf_recent_net_pnl"].to_numpy(dtype=np.float64)
    static_values = rows["static_net_pnl"].to_numpy(dtype=np.float64)
    trades = rows["trades"].to_numpy(dtype=np.float64)
    trade_weight = np.where(np.isfinite(trades) & (trades > 0), trades, 0.0)

    def weighted_rate(col: str) -> float:
        vals = rows[col].to_numpy(dtype=np.float64)
        mask = np.isfinite(vals) & (trade_weight > 0)
        if not mask.any():
            return float("nan")
        return float(np.average(vals[mask], weights=trade_weight[mask]))

    return {
        "run_id": run_id,
        **cfg.__dict__,
        "weeks": int(len(rows)),
        "fallback_weeks": int(rows["fallback_to_static"].sum()),
        "sum_net_pnl": float(np.sum(values)),
        "avg_week_net_pnl": float(np.mean(values)),
        "q20_week_net_pnl": float(np.quantile(values, 0.20)),
        "q35_week_net_pnl": float(np.quantile(values, 0.35)),
        "worst_week_net_pnl": float(np.min(values)),
        "objective": _objective(values, q35_weight, q20_weight),
        "wf_recent_sum_net_pnl": float(np.sum(wf_values)),
        "wf_recent_objective": _objective(wf_values, q35_weight, q20_weight),
        "static_sum_net_pnl": float(np.sum(static_values)),
        "static_objective": _objective(static_values, q35_weight, q20_weight),
        "delta_sum_net_pnl_vs_wf_recent": float(np.sum(values - wf_values)),
        "delta_objective_vs_wf_recent": _objective(values, q35_weight, q20_weight)
        - _objective(wf_values, q35_weight, q20_weight),
        "delta_worst_week_vs_wf_recent": float(np.min(values) - np.min(wf_values)),
        "positive_delta_weeks_vs_wf_recent": int(np.sum((values - wf_values) > 0.0)),
        "trade_weighted_full_sl_rate": weighted_rate("full_sl_rate"),
        "trade_weighted_timeout_rate": weighted_rate("timeout_rate"),
        "trade_weighted_hit_rate": weighted_rate("hit_rate"),
    }


def _run_grid(
    weekly_path: Path,
    signals_path: Path,
    q35_weight: float,
    q20_weight: float,
    start_week: str | None = None,
    end_week: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    weekly = _read_weekly(weekly_path)
    signals = _load_signals(signals_path)
    wide = _wide_weekly(weekly)
    configs = _configs()
    base_cache: dict[tuple[str, int, int], pd.DataFrame] = {}
    for cfg in configs:
        key = (cfg.feature_set, cfg.k_neighbors, cfg.min_train_weeks)
        if key not in base_cache:
            base_cache[key] = _precompute_neighbor_base(
                wide,
                signals,
                cfg.feature_set,
                cfg.k_neighbors,
                cfg.min_train_weeks,
                start_week=start_week,
                end_week=end_week,
            )
    summaries: list[dict[str, object]] = []
    runs: list[pd.DataFrame] = []
    for idx, cfg in enumerate(configs):
        rows = _evaluate_cached(base_cache[(cfg.feature_set, cfg.k_neighbors, cfg.min_train_weeks)], cfg)
        if rows.empty:
            continue
        rows["run_id"] = idx
        summaries.append(_summarize(rows, cfg, idx, q35_weight, q20_weight))
        runs.append(rows)
    summary = pd.DataFrame(summaries)
    if summary.empty:
        return summary, pd.DataFrame()
    summary = summary.sort_values(
        ["delta_objective_vs_wf_recent", "delta_sum_net_pnl_vs_wf_recent", "delta_worst_week_vs_wf_recent"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    all_runs = pd.concat(runs, ignore_index=True)
    return summary, all_runs


def _evaluate_config_on_weekly(
    weekly_path: Path,
    signals_path: Path,
    cfg: GuardConfig,
    q35_weight: float,
    q20_weight: float,
    start_week: str | None = None,
    end_week: str | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    weekly = _read_weekly(weekly_path)
    signals = _load_signals(signals_path)
    wide = _wide_weekly(weekly)
    rows = _evaluate(wide, signals, cfg, q35_weight, q20_weight, start_week=start_week, end_week=end_week)
    summary = _summarize(rows, cfg, -1, q35_weight, q20_weight) if not rows.empty else {}
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--development-weekly",
        type=Path,
        default=Path(
            "data_perp/reports/contextual_tp_sl_materialized_comparison_q35w07_q20w03_6mo_20260701/materialized_replay_week_comparison.csv"
        ),
    )
    parser.add_argument(
        "--frozen-weekly",
        type=Path,
        default=Path(
            "data_perp/reports/contextual_tp_sl_frozen_validation_may03_jun28_q35w06_q20w025_20260701/comparison/materialized_replay_week_comparison.csv"
        ),
    )
    parser.add_argument(
        "--signals",
        type=Path,
        default=Path("data_perp/reports/contextual_tp_sl_weekly_state_allocator_20260701/weekly_state_allocator_diagnostic_signals.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data_perp/reports/contextual_tp_sl_wfrecent_tail_execution_guard_20260701"))
    parser.add_argument("--q35-weight", type=float, default=0.70)
    parser.add_argument("--q20-weight", type=float, default=0.30)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary, all_runs = _run_grid(args.development_weekly, args.signals, args.q35_weight, args.q20_weight)
    if summary.empty:
        raise RuntimeError("No guard configurations produced evaluable rows")
    best = summary.iloc[0].to_dict()
    triggered_summary = summary[summary["fallback_weeks"] > 0].copy()
    best_triggered = triggered_summary.iloc[0].to_dict() if not triggered_summary.empty else {}
    best_tail_triggered = (
        triggered_summary.sort_values(
            ["trade_weighted_full_sl_rate", "trade_weighted_timeout_rate", "delta_objective_vs_wf_recent"],
            ascending=[True, True, False],
        )
        .iloc[0]
        .to_dict()
        if not triggered_summary.empty
        else {}
    )
    best_run_id = int(best["run_id"])
    best_rows = all_runs[all_runs["run_id"].eq(best_run_id)].copy()
    best_cfg = GuardConfig(
        feature_set=str(best["feature_set"]),
        k_neighbors=int(best["k_neighbors"]),
        min_neighbor_count=int(best["min_neighbor_count"]),
        min_train_weeks=int(best["min_train_weeks"]),
        full_sl_gate=float(best["full_sl_gate"]),
        timeout_gate=float(best["timeout_gate"]),
        min_net_delta=float(best["min_net_delta"]),
        min_q20_delta=float(best["min_q20_delta"]),
        min_positive_share=float(best["min_positive_share"]),
        mode=str(best["mode"]),
    )
    frozen_summary, frozen_rows = _evaluate_config_on_weekly(
        args.frozen_weekly,
        args.signals,
        best_cfg,
        args.q35_weight,
        args.q20_weight,
    )

    summary.to_csv(args.output_dir / "wfrecent_tail_execution_guard_grid.csv", index=False)
    best_rows.to_csv(args.output_dir / "wfrecent_tail_execution_guard_best_weeks.csv", index=False)
    if frozen_rows.empty:
        frozen_rows = pd.DataFrame()
    frozen_rows.to_csv(args.output_dir / "wfrecent_tail_execution_guard_frozen_weeks.csv", index=False)
    pd.DataFrame([frozen_summary]).to_csv(args.output_dir / "wfrecent_tail_execution_guard_frozen_summary.csv", index=False)
    manifest = {
        "development_weekly": str(args.development_weekly),
        "frozen_weekly": str(args.frozen_weekly),
        "signals": str(args.signals),
        "q35_weight": args.q35_weight,
        "q20_weight": args.q20_weight,
        "best_run_id": best_run_id,
        "grid_runs": int(len(summary)),
        "best_config": best_cfg.__dict__,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    lines = [
        "# wf_recent Tail/Execution Guard",
        "",
        "This is a lightweight walk-forward guard over existing materialized weekly replay outputs. It does not rerun the portfolio.",
        "",
        "Default action is `wf_recent`; the guard may fall back to `static` when prior similar diagnostic states suggest `wf_recent` will worsen full-SL/timeout behavior without sufficient PnL compensation.",
        "",
        "Diagnostic families searched: drift, recent hit-rate surprise, OOD, uncertainty, recent-HR+uncertainty, and all diagnostics.",
        "",
        "## Development Best Configuration",
        "",
        _fmt_table(
            pd.DataFrame([best]),
            [
                "run_id",
                "feature_set",
                "k_neighbors",
                "min_train_weeks",
                "mode",
                "full_sl_gate",
                "timeout_gate",
                "min_net_delta",
                "min_q20_delta",
                "min_positive_share",
                "fallback_weeks",
                "delta_sum_net_pnl_vs_wf_recent",
                "delta_objective_vs_wf_recent",
                "delta_worst_week_vs_wf_recent",
                "trade_weighted_full_sl_rate",
                "trade_weighted_timeout_rate",
                "trade_weighted_hit_rate",
            ],
        ),
        "",
        "## Development Weekly Decisions",
        "",
        _fmt_table(
            best_rows,
            [
                "week",
                "selected_label",
                "fallback_to_static",
                "net_pnl",
                "wf_recent_net_pnl",
                "static_net_pnl",
                "delta_vs_wf_recent_net_pnl",
                "full_sl_rate",
                "timeout_rate",
                "mean_delta_full_sl_rate",
                "mean_delta_timeout_rate",
                "mean_delta_net_pnl",
            ],
        ),
        "",
        "## Frozen May 3-June 28 Application",
        "",
        "The same selected guard was applied to the frozen May 3-June 28 weekly comparison table as a secondary validation proxy.",
        "",
        _fmt_table(
            pd.DataFrame([frozen_summary]) if frozen_summary else pd.DataFrame(),
            [
                "weeks",
                "fallback_weeks",
                "sum_net_pnl",
                "wf_recent_sum_net_pnl",
                "static_sum_net_pnl",
                "delta_sum_net_pnl_vs_wf_recent",
                "delta_objective_vs_wf_recent",
                "delta_worst_week_vs_wf_recent",
                "trade_weighted_full_sl_rate",
                "trade_weighted_timeout_rate",
                "trade_weighted_hit_rate",
            ],
        ),
        "",
        "Frozen weekly decisions:",
        "",
        _fmt_table(
            frozen_rows,
            [
                "week",
                "selected_label",
                "fallback_to_static",
                "net_pnl",
                "wf_recent_net_pnl",
                "static_net_pnl",
                "delta_vs_wf_recent_net_pnl",
                "full_sl_rate",
                "timeout_rate",
            ],
        ),
        "",
        "## Top Grid Runs",
        "",
        _fmt_table(
            summary,
            [
                "run_id",
                "feature_set",
                "k_neighbors",
                "min_train_weeks",
                "mode",
                "fallback_weeks",
                "delta_sum_net_pnl_vs_wf_recent",
                "delta_objective_vs_wf_recent",
                "delta_worst_week_vs_wf_recent",
                "trade_weighted_full_sl_rate",
                "trade_weighted_timeout_rate",
            ],
            max_rows=20,
        ),
        "",
        "## Triggered Rules",
        "",
        "The best overall rule did not trigger. The tables below show the least-bad triggered rule and the triggered rule with the lowest full-SL rate.",
        "",
        "Best triggered rule by objective:",
        "",
        _fmt_table(
            pd.DataFrame([best_triggered]) if best_triggered else pd.DataFrame(),
            [
                "run_id",
                "feature_set",
                "k_neighbors",
                "min_train_weeks",
                "mode",
                "fallback_weeks",
                "delta_sum_net_pnl_vs_wf_recent",
                "delta_objective_vs_wf_recent",
                "delta_worst_week_vs_wf_recent",
                "trade_weighted_full_sl_rate",
                "trade_weighted_timeout_rate",
            ],
        ),
        "",
        "Lowest full-SL triggered rule:",
        "",
        _fmt_table(
            pd.DataFrame([best_tail_triggered]) if best_tail_triggered else pd.DataFrame(),
            [
                "run_id",
                "feature_set",
                "k_neighbors",
                "min_train_weeks",
                "mode",
                "fallback_weeks",
                "delta_sum_net_pnl_vs_wf_recent",
                "delta_objective_vs_wf_recent",
                "delta_worst_week_vs_wf_recent",
                "trade_weighted_full_sl_rate",
                "trade_weighted_timeout_rate",
            ],
        ),
    ]
    (args.output_dir / "wfrecent_tail_execution_guard_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
