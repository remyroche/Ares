#!/usr/bin/env python3
"""Evaluate geometry scores under the deployed 8-day reachable-EV policy logic.

The evaluation keeps the current global top-10 activity count, but determines
admission order from side/archetype-local score distributions and reachable EV.
Only outcomes old enough to have resolved are available to each decision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

KEYS = ["__ts__", "__symbol__", "side_name", "archetype_policy_key", "evaluation_scope"]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _target_ev(
    prefix_sum: np.ndarray,
    prefix_count: np.ndarray,
    cutoff: int,
    min_rows: int,
) -> float:
    count = int(prefix_count[cutoff])
    if count < int(min_rows):
        return np.nan
    return float(prefix_sum[cutoff] / count)


def _threshold_for_target(
    score: np.ndarray,
    ev: np.ndarray,
    target: float,
    min_rows: int,
) -> float:
    valid = np.isfinite(score) & np.isfinite(ev)
    score = score[valid]
    ev = ev[valid]
    if score.size < int(min_rows) or not np.isfinite(target):
        return np.nan
    grid = np.unique(np.nanquantile(score, np.linspace(0.70, 0.99, 60)))
    order = np.argsort(score, kind="stable")
    sorted_score = score[order]
    sorted_ev = ev[order]
    reverse_sum = np.cumsum(sorted_ev[::-1], dtype=np.float64)[::-1]
    reverse_count = np.arange(score.size, 0, -1, dtype=np.int64)
    best_threshold = np.nan
    best_gap = np.inf
    for threshold in grid:
        position = int(np.searchsorted(sorted_score, threshold, side="left"))
        count = int(reverse_count[position]) if position < score.size else 0
        if count < int(min_rows):
            continue
        mean_ev = float(reverse_sum[position] / count)
        gap = abs(mean_ev - target)
        if mean_ev >= target and gap < best_gap:
            best_threshold = float(threshold)
            best_gap = gap
    return (
        float(np.nanquantile(score, 0.99))
        if not np.isfinite(best_threshold)
        else best_threshold
    )


def _rank_against(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    reference = np.sort(reference[np.isfinite(reference)])
    output = np.full(values.size, np.nan, dtype=np.float32)
    finite = np.isfinite(values)
    if reference.size and finite.any():
        output[finite] = (
            np.searchsorted(reference, values[finite], side="right")
            / float(reference.size)
        ).astype(np.float32)
    return output


def _prefix_for_top10(
    ev: np.ndarray, rank: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(ev) & np.isfinite(rank) & (rank >= 0.90)
    values = np.where(valid, ev, 0.0)
    return (
        np.r_[0.0, np.cumsum(values, dtype=np.float64)],
        np.r_[0, np.cumsum(valid, dtype=np.int64)],
    )


def _local_prefixes(
    archetype: np.ndarray,
    ev: np.ndarray,
    rank: np.ndarray,
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    output: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for code in np.unique(archetype):
        positions = np.flatnonzero(archetype == code).astype(np.int64)
        prefix_sum, prefix_count = _prefix_for_top10(ev[positions], rank[positions])
        output[int(code)] = positions, prefix_sum, prefix_count
    return output


def _local_target(
    state: tuple[np.ndarray, np.ndarray, np.ndarray],
    cutoff: int,
    min_rows: int,
) -> float:
    positions, prefix_sum, prefix_count = state
    local_cutoff = int(np.searchsorted(positions, cutoff, side="left"))
    return _target_ev(prefix_sum, prefix_count, local_cutoff, min_rows)


def _select_timestamp(
    *,
    current: np.ndarray,
    recent: np.ndarray,
    cutoff: int,
    score: np.ndarray,
    ev: np.ndarray,
    base_rank: np.ndarray,
    archetype: np.ndarray,
    global_prefix: tuple[np.ndarray, np.ndarray],
    local_prefix: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    min_rows: int,
    arch_min_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    baseline_count = int(np.sum(base_rank[current] >= 0.90))
    if baseline_count <= 0 or recent.size < int(min_rows):
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
        )
    global_target = _target_ev(*global_prefix, cutoff, min_rows)
    global_threshold = _threshold_for_target(
        score[recent], ev[recent], global_target, min_rows
    )
    global_rank = _rank_against(score[current], score[recent])
    local_rank = global_rank.copy()
    thresholds = np.full(current.size, global_threshold, dtype=np.float32)
    targets = np.full(current.size, global_target, dtype=np.float32)
    eligible = np.zeros(current.size, dtype=bool)
    current_arch = archetype[current]
    recent_arch = archetype[recent]
    for code in np.unique(current_arch):
        current_mask = current_arch == code
        recent_local = recent[recent_arch == code]
        target = _local_target(local_prefix[int(code)], cutoff, arch_min_rows)
        threshold = _threshold_for_target(
            score[recent_local], ev[recent_local], target, arch_min_rows
        )
        if not np.isfinite(threshold):
            threshold = global_threshold
            target = global_target
            rank_reference = score[recent]
        else:
            rank_reference = score[recent_local]
        local_rank[current_mask] = _rank_against(
            score[current[current_mask]], rank_reference
        )
        thresholds[current_mask] = np.float32(threshold)
        targets[current_mask] = np.float32(target)
        eligible[current_mask] = score[current[current_mask]] >= threshold
    finite_rank = np.nan_to_num(local_rank, nan=-1.0)
    pool = np.flatnonzero(eligible)
    if pool.size < baseline_count:
        pool = np.arange(current.size, dtype=np.int64)
    count = min(baseline_count, pool.size)
    chosen_local = pool[
        np.argpartition(finite_rank[pool], pool.size - count)[pool.size - count :]
    ]
    order = np.argsort(finite_rank[chosen_local], kind="stable")[::-1]
    chosen_local = chosen_local[order]
    return current[chosen_local], thresholds[chosen_local], targets[chosen_local]


def _evaluate_policy(
    frame: pd.DataFrame,
    *,
    score_col: str,
    selector: str,
    eval_start: pd.Timestamp,
    window_days: int,
    outcome_embargo_hours: int,
    min_rows: int,
    arch_min_rows: int,
) -> pd.DataFrame:
    timestamp = frame["__ts__"].astype("int64").to_numpy()
    score = pd.to_numeric(frame[score_col], errors="coerce").to_numpy(dtype=np.float32)
    ev = pd.to_numeric(frame["ev_after_1pct"], errors="coerce").to_numpy(
        dtype=np.float32
    )
    base_rank = pd.to_numeric(frame["base_batch_rank"], errors="coerce").to_numpy(
        dtype=np.float32
    )
    archetype, _ = pd.factorize(frame["policy_archetype"], sort=True)
    archetype = archetype.astype(np.int16)
    global_prefix = _prefix_for_top10(ev, base_rank)
    local_prefix = _local_prefixes(archetype, ev, base_rank)
    unique_ts, starts = np.unique(timestamp, return_index=True)
    ends = np.r_[starts[1:], len(frame)]
    eval_ns = int(eval_start.value)
    window_ns = int(pd.Timedelta(days=window_days).value)
    embargo_ns = int(pd.Timedelta(hours=outcome_embargo_hours).value)
    selected_parts: list[pd.DataFrame] = []
    for ts_ns, start, end in zip(unique_ts, starts, ends, strict=False):
        if int(ts_ns) < eval_ns:
            continue
        outcome_cutoff_ns = int(ts_ns) - embargo_ns
        cutoff = int(np.searchsorted(timestamp, outcome_cutoff_ns, side="right"))
        recent_start = int(
            np.searchsorted(timestamp, outcome_cutoff_ns - window_ns, side="left")
        )
        current = np.arange(int(start), int(end), dtype=np.int64)
        recent = np.arange(recent_start, cutoff, dtype=np.int64)
        chosen, thresholds, targets = _select_timestamp(
            current=current,
            recent=recent,
            cutoff=cutoff,
            score=score,
            ev=ev,
            base_rank=base_rank,
            archetype=archetype,
            global_prefix=global_prefix,
            local_prefix=local_prefix,
            min_rows=min_rows,
            arch_min_rows=arch_min_rows,
        )
        if not chosen.size:
            continue
        part = frame.iloc[chosen].copy()
        part["selector"] = selector
        part["dynamic_score_threshold"] = thresholds
        part["dynamic_ev_target"] = targets
        selected_parts.append(part)
    return (
        pd.concat(selected_parts, ignore_index=True)
        if selected_parts
        else frame.iloc[0:0].copy()
    )


def _fixed_top10(
    frame: pd.DataFrame, rank_col: str, selector: str, eval_start: pd.Timestamp
) -> pd.DataFrame:
    output = frame.loc[
        frame["__ts__"].ge(eval_start)
        & pd.to_numeric(frame[rank_col], errors="coerce").ge(0.90)
    ].copy()
    output["selector"] = selector
    return output


def _fixed_matched_top10(
    frame: pd.DataFrame,
    score_col: str,
    selector: str,
    eval_start: pd.Timestamp,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    evaluation = frame.loc[frame["__ts__"].ge(eval_start)]
    for _, part in evaluation.groupby("__ts__", sort=False):
        count = int(
            pd.to_numeric(part["base_batch_rank"], errors="coerce").ge(0.90).sum()
        )
        if count <= 0:
            continue
        score = pd.to_numeric(part[score_col], errors="coerce").to_numpy(
            dtype=np.float32
        )
        count = min(count, len(part))
        positions = np.argpartition(score, len(part) - count)[len(part) - count :]
        parts.append(part.iloc[positions])
    output = (
        pd.concat(parts, ignore_index=True) if parts else evaluation.iloc[0:0].copy()
    )
    output["selector"] = selector
    return output


def _metrics(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    grouped = frame.groupby(
        ["selector", *groups], observed=True, dropna=False, sort=True
    )
    return grouped.agg(
        selected_rows=("ev_after_1pct", "size"),
        mean_ev_after_1pct=("ev_after_1pct", "mean"),
        sum_ev_after_1pct=("ev_after_1pct", "sum"),
        clean_exec_precision=("clean_exec", "mean"),
        full_path_bad_mae_rate=("full_path_bad_mae_1r", "mean"),
        timeout_rate=("timeout", "mean"),
        symbols=("__symbol__", "nunique"),
    ).reset_index()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ablation-predictions",
        type=Path,
        default=Path(
            "data_perp/reports/meta_cross_sectional_geometry_ablation_20260711_v1/cross_sectional_geometry_predictions.parquet"
        ),
    )
    parser.add_argument(
        "--balanced-predictions",
        type=Path,
        default=Path(
            "data_perp/reports/meta_geometry_rank_nudge_20260711_v1/balanced_composite_predictions.parquet"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_perp/reports/meta_geometry_reachable_ev_policy_20260711_v1"),
    )
    parser.add_argument("--eval-start", default="2026-05-01")
    parser.add_argument("--window-days", type=int, default=8)
    parser.add_argument("--outcome-embargo-hours", type=int, default=12)
    parser.add_argument("--min-reference-rows", type=int, default=40)
    parser.add_argument("--arch-min-reference-rows", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = pd.read_parquet(args.ablation_predictions)
    balanced = pd.read_parquet(args.balanced_predictions)
    source = source.merge(
        balanced[
            KEYS + ["score_balanced_composite", "selected_top10_balanced_composite"]
        ],
        on=KEYS,
        how="inner",
        validate="one_to_one",
    )
    source["__ts__"] = pd.to_datetime(source["__ts__"], utc=True, errors="coerce")
    source = source.sort_values(
        ["__ts__", "__symbol__", "side_name"], kind="stable"
    ).reset_index(drop=True)
    source["policy_archetype"] = (
        source["side_name"].astype(str)
        + "__"
        + source["archetype_policy_key"].astype(str)
    )
    source["balanced_batch_rank"] = (
        source.groupby("__ts__", sort=False)["score_balanced_composite"]
        .rank(method="average", pct=True)
        .astype(np.float32)
    )
    source["month"] = source["__ts__"].dt.strftime("%Y-%m")
    source["day"] = source["__ts__"].dt.floor("D")
    source["week_start"] = source["day"] - pd.to_timedelta(
        source["day"].dt.weekday, unit="D"
    )
    eval_start = pd.Timestamp(args.eval_start)
    eval_start = (
        eval_start.tz_localize("UTC")
        if eval_start.tzinfo is None
        else eval_start.tz_convert("UTC")
    )
    selected = [
        _fixed_top10(source, "base_batch_rank", "baseline_fixed_top10", eval_start),
        _fixed_matched_top10(
            source,
            "score_balanced_composite",
            "geometry_fixed_top10",
            eval_start,
        ),
        _evaluate_policy(
            source,
            score_col="base_batch_rank",
            selector="baseline_reachable_ev_8d",
            eval_start=eval_start,
            window_days=args.window_days,
            outcome_embargo_hours=args.outcome_embargo_hours,
            min_rows=args.min_reference_rows,
            arch_min_rows=args.arch_min_reference_rows,
        ),
        _evaluate_policy(
            source,
            score_col="score_balanced_composite",
            selector="geometry_reachable_ev_8d",
            eval_start=eval_start,
            window_days=args.window_days,
            outcome_embargo_hours=args.outcome_embargo_hours,
            min_rows=args.min_reference_rows,
            arch_min_rows=args.arch_min_reference_rows,
        ),
    ]
    chosen = pd.concat(selected, ignore_index=True, sort=False)
    chosen.to_parquet(
        args.output_dir / "selected_predictions.parquet",
        index=False,
        compression="zstd",
    )
    scopes = {
        "overall": [],
        "month": ["month"],
        "week": ["week_start"],
        "day": ["day"],
        "side": ["side_name"],
        "archetype": ["side_name", "archetype_policy_key"],
        "month_side_archetype": ["month", "side_name", "archetype_policy_key"],
    }
    for name, groups in scopes.items():
        _metrics(chosen, groups).to_csv(
            args.output_dir / f"metrics_{name}.csv", index=False
        )
    manifest = {
        "schema": "meta_geometry_reachable_ev_policy_v1",
        "policy_reference": "ev_target_archetype_reachable_match_current_activity_8d_hr_off",
        "score_contract": "rank-space baseline versus balanced geometry score",
        "activity_contract": "exact current global within-timestamp top-10 count",
        "cost_contract": "ev_after_1pct includes 1% round-trip cost",
        "window_days": args.window_days,
        "outcome_embargo_hours": args.outcome_embargo_hours,
        "eval_start": eval_start,
        "eval_end": source["__ts__"].max(),
        "leakage_contract": "Every threshold and EV target uses only rows ending at least outcome_embargo_hours before the decision timestamp.",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True), encoding="utf-8"
    )
    print(_metrics(chosen, []).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
