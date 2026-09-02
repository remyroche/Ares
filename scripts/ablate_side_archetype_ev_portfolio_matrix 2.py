#!/usr/bin/env python3
"""Ablate fixed side-archetype EV targets and robust 28d-style calibration.

Every arm uses the same frozen OOF model rows and hierarchical EV map. Daily
realized-minus-mapped residual corrections are estimated causally from resolved
outcomes, with optional symmetric trimming of robust-IQR-normalized daily
residual states. A compact deterministic portfolio replay then limits new
entries and concurrent positions.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from scripts.evaluate_side_archetype_expected_ev_policy import _load_rows


DEFAULT_TARGETS = (0.007, 0.008, 0.009)
DEFAULT_TRIMS = (0.0, 0.10, 0.15, 0.20, 0.25)
DEFAULT_PERIODS = (14, 21, 28)


@dataclass(frozen=True)
class Arm:
    target_ev: float
    trim_fraction: float
    period_days: int

    @property
    def name(self) -> str:
        target_bps = int(round(self.target_ev * 10_000.0))
        trim_pct = int(round(self.trim_fraction * 100.0))
        return f"ev{target_bps}bps_trim{trim_pct:02d}_period{self.period_days}d"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _daily_stats(
    frame: pd.DataFrame, groups: list[str]
) -> pd.DataFrame:
    keys = [*groups, "outcome_day"]
    return (
        frame.groupby(keys, sort=False, observed=True)["residual"]
        .agg(["sum", "count", "mean"])
        .reset_index()
    )


def _trimmed_correction(
    stats: pd.DataFrame, trim_fraction: float
) -> tuple[float, int, int, float]:
    """Return row-weighted correction after symmetric robust daily trimming."""
    if stats.empty:
        return np.nan, 0, 0, np.nan
    daily = pd.to_numeric(stats["mean"], errors="coerce").to_numpy(
        dtype=np.float64, copy=False
    )
    sums = pd.to_numeric(stats["sum"], errors="coerce").to_numpy(
        dtype=np.float64, copy=False
    )
    counts = pd.to_numeric(stats["count"], errors="coerce").to_numpy(
        dtype=np.float64, copy=False
    )
    finite = np.isfinite(daily) & np.isfinite(sums) & np.isfinite(counts) & (counts > 0)
    if not finite.any():
        return np.nan, 0, 0, np.nan
    daily, sums, counts = daily[finite], sums[finite], counts[finite]
    median = float(np.median(daily))
    q25, q75 = np.quantile(daily, [0.25, 0.75])
    iqr = max(float(q75 - q25), 1e-8)
    robust_z = (daily - median) / iqr
    keep = np.ones(len(daily), dtype=bool)
    trim = float(np.clip(trim_fraction, 0.0, 0.49))
    if trim > 0.0 and len(daily) >= 4:
        low, high = np.quantile(robust_z, [trim, 1.0 - trim])
        keep = (robust_z >= low) & (robust_z <= high)
    if not keep.any():
        return np.nan, 0, int(len(daily)), iqr
    support = int(np.sum(counts[keep]))
    correction = float(np.sum(sums[keep]) / max(np.sum(counts[keep]), 1.0))
    return correction, support, int(keep.sum()), iqr


def _group_corrections(
    stats: pd.DataFrame,
    groups: list[str],
    *,
    start_day: pd.Timestamp,
    end_day: pd.Timestamp,
    trim_fraction: float,
) -> dict[Any, tuple[float, int, int, float]]:
    window = stats.loc[
        stats["outcome_day"].ge(start_day)
        & stats["outcome_day"].lt(end_day)
    ]
    if window.empty:
        return {}
    if not groups:
        return {"__global__": _trimmed_correction(window, trim_fraction)}
    group_arg: str | list[str] = groups[0] if len(groups) == 1 else groups
    result: dict[Any, tuple[float, int, int, float]] = {}
    for key, part in window.groupby(group_arg, sort=False, observed=True):
        normalized_key = str(key) if len(groups) == 1 else tuple(str(v) for v in key)
        result[normalized_key] = _trimmed_correction(part, trim_fraction)
    return result


def _corrected_ev_for_arm(
    source: pd.DataFrame,
    arm: Arm,
    *,
    global_daily: pd.DataFrame,
    side_daily: pd.DataFrame,
    local_daily: pd.DataFrame,
    side_support_target: float = 320.0,
    local_support_target: float = 160.0,
    correction_cap: float = 0.03,
) -> tuple[np.ndarray, pd.DataFrame]:
    mapped = pd.to_numeric(
        source["expected_net_ev_after_1pct_mlp_direct"], errors="coerce"
    ).to_numpy(dtype=np.float64, copy=False)
    side = source["side_name"].astype(str).to_numpy(copy=False)
    archetype = source["policy_archetype"].astype(str).to_numpy(copy=False)
    entry_day = pd.to_datetime(source["__ts__"], utc=True).dt.floor("D")
    corrected = mapped.copy()
    diagnostics: list[dict[str, Any]] = []

    day_groups = entry_day.groupby(entry_day, sort=True).groups
    for day, index_values in day_groups.items():
        asof = pd.Timestamp(day)
        start = asof - pd.Timedelta(days=arm.period_days)
        global_stats = _group_corrections(
            global_daily,
            [],
            start_day=start,
            end_day=asof,
            trim_fraction=arm.trim_fraction,
        ).get("__global__", (0.0, 0, 0, np.nan))
        global_correction = float(
            np.clip(
                global_stats[0] if np.isfinite(global_stats[0]) else 0.0,
                -correction_cap,
                correction_cap,
            )
        )
        side_stats = _group_corrections(
            side_daily,
            ["side_name"],
            start_day=start,
            end_day=asof,
            trim_fraction=arm.trim_fraction,
        )
        local_stats = _group_corrections(
            local_daily,
            ["side_name", "policy_archetype"],
            start_day=start,
            end_day=asof,
            trim_fraction=arm.trim_fraction,
        )
        side_correction: dict[str, tuple[float, int]] = {}
        for side_key, stats in side_stats.items():
            value, support = stats[0], stats[1]
            alpha = float(
                np.clip(support / max(side_support_target, 1.0), 0.0, 1.0)
            )
            local_value = value if np.isfinite(value) else global_correction
            shrunk = (1.0 - alpha) * global_correction + alpha * local_value
            side_correction[str(side_key)] = (
                float(np.clip(shrunk, -correction_cap, correction_cap)),
                int(support),
            )
        local_correction: dict[tuple[str, str], tuple[float, int]] = {}
        for key, stats in local_stats.items():
            side_key, arch_key = str(key[0]), str(key[1])
            parent = side_correction.get(
                side_key, (global_correction, global_stats[1])
            )[0]
            value, support = stats[0], stats[1]
            alpha = float(
                np.clip(support / max(local_support_target, 1.0), 0.0, 1.0)
            )
            local_value = value if np.isfinite(value) else parent
            shrunk = (1.0 - alpha) * parent + alpha * local_value
            local_correction[(side_key, arch_key)] = (
                float(np.clip(shrunk, -correction_cap, correction_cap)),
                int(support),
            )

        indices = np.asarray(list(index_values), dtype=np.int64)
        corrections = np.empty(len(indices), dtype=np.float64)
        local_count = 0
        for pos, row_idx in enumerate(indices):
            key = (str(side[row_idx]), str(archetype[row_idx]))
            local = local_correction.get(key)
            if local is not None:
                corrections[pos] = local[0]
                local_count += 1
            else:
                corrections[pos] = side_correction.get(
                    key[0], (global_correction, global_stats[1])
                )[0]
        corrected[indices] = mapped[indices] + corrections
        diagnostics.append(
            {
                "arm": arm.name,
                "calibration_asof": asof,
                "global_correction": global_correction,
                "global_support": int(global_stats[1]),
                "global_days_retained": int(global_stats[2]),
                "global_iqr": global_stats[3],
                "side_groups": int(len(side_correction)),
                "local_groups": int(len(local_correction)),
                "current_rows": int(len(indices)),
                "local_rows": int(local_count),
            }
        )
    return corrected.astype(np.float32), pd.DataFrame(diagnostics)


def _portfolio_replay(
    source: pd.DataFrame,
    corrected_ev: np.ndarray,
    *,
    target_ev: float,
    max_new_entries_per_bar: int,
    max_concurrent_positions: int,
    outcome_horizon_hours: int,
) -> np.ndarray:
    ts_ns = pd.to_datetime(source["__ts__"], utc=True).astype("int64").to_numpy(
        dtype=np.int64, copy=False
    )
    exit_ns = ts_ns + int(pd.Timedelta(hours=outcome_horizon_hours).value)
    symbols = source["__symbol__"].astype(str).to_numpy(copy=False)
    realized = pd.to_numeric(source["ev_after_1pct"], errors="coerce").to_numpy(
        dtype=np.float64, copy=False
    )
    parent_rank = pd.to_numeric(source["policy_parent_rank"], errors="coerce").to_numpy(
        dtype=np.float64, copy=False
    )
    _, starts, counts = np.unique(ts_ns, return_index=True, return_counts=True)
    open_positions: list[tuple[int, str]] = []
    selected: list[int] = []
    for start, count in zip(starts, counts):
        now = int(ts_ns[start])
        open_positions = [item for item in open_positions if item[0] > now]
        capacity = max(int(max_concurrent_positions) - len(open_positions), 0)
        if capacity <= 0:
            continue
        stop = int(start + count)
        idx = np.arange(start, stop, dtype=np.int64)
        finite = np.isfinite(corrected_ev[idx]) & np.isfinite(realized[idx])
        idx = idx[finite & (corrected_ev[idx] >= float(target_ev))]
        if idx.size == 0:
            continue
        parent = np.where(np.isfinite(parent_rank[idx]), parent_rank[idx], 0.0)
        order = np.lexsort((-parent, -corrected_ev[idx]))
        open_symbols = {item[1] for item in open_positions}
        admitted = 0
        for row_idx in idx[order]:
            symbol = str(symbols[row_idx])
            if symbol in open_symbols:
                continue
            selected.append(int(row_idx))
            open_positions.append((int(exit_ns[row_idx]), symbol))
            open_symbols.add(symbol)
            admitted += 1
            if admitted >= int(max_new_entries_per_bar) or admitted >= capacity:
                break
    return np.asarray(selected, dtype=np.int64)


def _portfolio_metrics(
    source: pd.DataFrame,
    selected_idx: np.ndarray,
    arm: Arm,
    *,
    max_concurrent_positions: int,
    outcome_horizon_hours: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    selected = source.iloc[selected_idx].copy()
    selected["arm"] = arm.name
    selected["target_ev"] = arm.target_ev
    selected["trim_fraction"] = arm.trim_fraction
    selected["period_days"] = arm.period_days
    selected["outcome_resolved_at"] = pd.to_datetime(
        selected["__ts__"], utc=True
    ) + pd.Timedelta(hours=outcome_horizon_hours)
    selected["slot_pnl"] = (
        pd.to_numeric(selected["ev_after_1pct"], errors="coerce")
        / max(float(max_concurrent_positions), 1.0)
    )
    entry_start = pd.to_datetime(source["__ts__"], utc=True).min().floor("D")
    entry_end = pd.to_datetime(source["__ts__"], utc=True).max().floor("D")
    pnl_end = (
        pd.to_datetime(source["__ts__"], utc=True).max()
        + pd.Timedelta(hours=outcome_horizon_hours)
    ).floor("D")
    pnl_days = pd.date_range(entry_start, pnl_end, freq="D", tz="UTC")
    daily = (
        selected.assign(pnl_day=selected["outcome_resolved_at"].dt.floor("D"))
        .groupby("pnl_day", observed=True)["slot_pnl"]
        .sum()
        .reindex(pnl_days, fill_value=0.0)
    )
    period_index = daily.index.tz_localize(None)
    weekly = daily.groupby(period_index.to_period("W-SUN")).sum()
    monthly = daily.groupby(period_index.to_period("M")).sum()
    ev = pd.to_numeric(selected["ev_after_1pct"], errors="coerce")
    entry_days = int((entry_end - entry_start).days + 1)
    metrics = {
        "arm": arm.name,
        "target_ev": arm.target_ev,
        "trim_fraction": arm.trim_fraction,
        "period_days": arm.period_days,
        "trades": int(len(selected)),
        "trades_per_day": float(len(selected) / max(entry_days, 1)),
        "net_ev_per_trade": float(ev.mean()) if len(ev) else np.nan,
        "avg_net_pnl_per_day": float(daily.mean()) if len(daily) else np.nan,
        "negative_pnl_day_rate": float((daily < 0.0).mean()) if len(daily) else np.nan,
        "worst_week_pnl": float(weekly.min()) if len(weekly) else np.nan,
        "worst_month_pnl": float(monthly.min()) if len(monthly) else np.nan,
        "total_slot_pnl": float(daily.sum()) if len(daily) else 0.0,
        "positive_week_rate": float((weekly > 0.0).mean()) if len(weekly) else np.nan,
        "entry_calendar_days": entry_days,
        "pnl_calendar_days": int(len(daily)),
    }
    return metrics, selected


def _validate_portfolio_constraints(trades: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for arm, group in trades.groupby("arm", sort=False, observed=True):
        group = group.sort_values("__ts__", kind="stable")
        max_new = int(group.groupby("__ts__", observed=True).size().max())
        open_positions: list[tuple[pd.Timestamp, str]] = []
        max_open = 0
        duplicate_overlap = False
        for timestamp, batch in group.groupby("__ts__", sort=True, observed=True):
            now = pd.Timestamp(timestamp)
            open_positions = [item for item in open_positions if item[0] > now]
            active_symbols = {item[1] for item in open_positions}
            for exit_ts, symbol in zip(
                batch["outcome_resolved_at"], batch["__symbol__"]
            ):
                symbol_text = str(symbol)
                duplicate_overlap |= symbol_text in active_symbols
                open_positions.append((pd.Timestamp(exit_ts), symbol_text))
                active_symbols.add(symbol_text)
            max_open = max(max_open, len(open_positions))
        rows.append(
            {
                "arm": str(arm),
                "max_new_entries_observed": max_new,
                "max_concurrent_positions_observed": max_open,
                "same_symbol_overlap_observed": bool(duplicate_overlap),
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oos-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", default="2026-04-01T00:00:00Z")
    parser.add_argument("--targets", type=float, nargs="+", default=list(DEFAULT_TARGETS))
    parser.add_argument("--trims", type=float, nargs="+", default=list(DEFAULT_TRIMS))
    parser.add_argument("--periods", type=int, nargs="+", default=list(DEFAULT_PERIODS))
    parser.add_argument("--max-new-entries-per-bar", type=int, default=2)
    parser.add_argument("--max-concurrent-positions", type=int, default=8)
    parser.add_argument("--outcome-horizon-hours", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(args.start)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    source = _load_rows(args.oos_predictions, start).reset_index(drop=True)
    source["outcome_day"] = (
        pd.to_datetime(source["__ts__"], utc=True)
        + pd.Timedelta(hours=args.outcome_horizon_hours)
    ).dt.floor("D")
    source["residual"] = (
        pd.to_numeric(source["ev_after_1pct"], errors="coerce")
        - pd.to_numeric(
            source["expected_net_ev_after_1pct_mlp_direct"], errors="coerce"
        )
    )
    finite = np.isfinite(source["residual"].to_numpy(dtype=np.float64, copy=False))
    residual_rows = source.loc[
        finite, ["outcome_day", "side_name", "policy_archetype", "residual"]
    ]
    global_daily = _daily_stats(residual_rows, [])
    side_daily = _daily_stats(residual_rows, ["side_name"])
    local_daily = _daily_stats(
        residual_rows, ["side_name", "policy_archetype"]
    )

    targets = sorted(set(float(value) for value in args.targets))
    trims = sorted(set(float(value) for value in args.trims))
    periods = sorted(set(int(value) for value in args.periods))
    metrics_rows: list[dict[str, Any]] = []
    trade_parts: list[pd.DataFrame] = []
    calibration_parts: list[pd.DataFrame] = []
    completed = 0
    total = len(targets) * len(trims) * len(periods)
    for period in periods:
        for trim in trims:
            template = Arm(targets[0], trim, period)
            corrected_ev, calibration = _corrected_ev_for_arm(
                source,
                template,
                global_daily=global_daily,
                side_daily=side_daily,
                local_daily=local_daily,
            )
            calibration["trim_fraction"] = trim
            calibration["period_days"] = period
            calibration_parts.append(calibration)
            for target in targets:
                arm = Arm(target, trim, period)
                selected_idx = _portfolio_replay(
                    source,
                    corrected_ev,
                    target_ev=target,
                    max_new_entries_per_bar=args.max_new_entries_per_bar,
                    max_concurrent_positions=args.max_concurrent_positions,
                    outcome_horizon_hours=args.outcome_horizon_hours,
                )
                metrics, trades = _portfolio_metrics(
                    source,
                    selected_idx,
                    arm,
                    max_concurrent_positions=args.max_concurrent_positions,
                    outcome_horizon_hours=args.outcome_horizon_hours,
                )
                trades["corrected_expected_ev"] = corrected_ev[selected_idx]
                metrics_rows.append(metrics)
                trade_parts.append(trades)
                completed += 1
                print(
                    f"completed {completed}/{total}: {arm.name} "
                    f"trades={len(trades):,} ev={metrics['net_ev_per_trade']:.6f}",
                    flush=True,
                )

    metrics = pd.DataFrame(metrics_rows).sort_values(
        ["net_ev_per_trade", "worst_week_pnl", "avg_net_pnl_per_day"],
        ascending=False,
        kind="stable",
    )
    metrics.to_csv(args.output_dir / "portfolio_matrix_metrics.csv", index=False)
    all_trades = pd.concat(trade_parts, ignore_index=True, copy=False)
    all_trades.to_parquet(
        args.output_dir / "portfolio_selected_trades.parquet",
        index=False,
        compression="zstd",
    )
    constraint_validation = _validate_portfolio_constraints(all_trades)
    constraint_validation.to_csv(
        args.output_dir / "portfolio_constraint_validation.csv", index=False
    )
    if (
        constraint_validation["max_new_entries_observed"].gt(
            args.max_new_entries_per_bar
        ).any()
        or constraint_validation["max_concurrent_positions_observed"].gt(
            args.max_concurrent_positions
        ).any()
        or constraint_validation["same_symbol_overlap_observed"].any()
    ):
        raise AssertionError("portfolio constraint validation failed")
    pd.concat(calibration_parts, ignore_index=True, copy=False).to_parquet(
        args.output_dir / "calibration_diagnostics.parquet",
        index=False,
        compression="zstd",
    )
    manifest = {
        "schema": "side_archetype_ev_robust_portfolio_matrix_v1",
        "source": str(args.oos_predictions),
        "rows": int(len(source)),
        "evaluation_start": source["__ts__"].min().isoformat(),
        "evaluation_end": source["__ts__"].max().isoformat(),
        "targets": targets,
        "symmetric_trim_fractions": trims,
        "period_days": periods,
        "arms": int(total),
        "portfolio": {
            "max_new_entries_per_bar": int(args.max_new_entries_per_bar),
            "max_concurrent_positions": int(args.max_concurrent_positions),
            "same_symbol_overlap_allowed": False,
            "holding_period_hours": int(args.outcome_horizon_hours),
            "position_weight": 1.0 / max(float(args.max_concurrent_positions), 1.0),
        },
        "robust_contract": (
            "Within each causal lookback and side x archetype, daily residual "
            "means are median/IQR normalized. The requested lower and upper "
            "fractions are removed symmetrically, then the correction is "
            "recomputed from all rows on retained days and shrunk local -> "
            "side -> global."
        ),
        "cost_contract": (
            "ev_after_1pct is net of the sole 1% round-trip cost. No additional "
            "fee or spread is subtracted in this matrix."
        ),
        "metric_contract": (
            "net_ev_per_trade is notional return/trade. Daily, weekly, and "
            "monthly PnL use equal 1/8 portfolio slots and are attributed to "
            "the 12h outcome-resolution day; zero-PnL calendar days are retained."
        ),
    }
    _write_json(args.output_dir / "manifest.json", manifest)
    print(metrics.head(15).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
