#!/usr/bin/env python3
"""Compare exact-V9 base and new MLP meta scores on identical OOS rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


KEYS = ["__ts__", "__symbol__", "side_name", "archetype_policy_key"]
OUTCOMES = [
    "ev_after_1pct",
    "clean_exec",
    "dirty_positive",
    "full_path_bad_mae_1r",
    "timeout",
]
SCORES = {
    "base": "score_base",
    "meta_lgbm": "hit_probability",
    "meta_v9_mlp": "expected_ev_rank_score",
}


def _top_fraction_mask(values: pd.Series, fraction: float) -> np.ndarray:
    score = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    valid = np.flatnonzero(np.isfinite(score))
    mask = np.zeros(len(score), dtype=bool)
    if not len(valid):
        return mask
    count = max(1, int(np.ceil(float(fraction) * len(valid))))
    chosen = valid[np.argpartition(score[valid], -count)[-count:]]
    mask[chosen] = True
    return mask


def _metrics(rows: pd.DataFrame, mask: np.ndarray) -> dict[str, float | int]:
    selected = rows.loc[mask]
    ev = pd.to_numeric(selected["ev_after_1pct"], errors="coerce")
    return {
        "candidate_rows": int(len(rows)),
        "selected_rows": int(len(selected)),
        "mean_net_ev_after_1pct": float(ev.mean()) if len(ev) else np.nan,
        "sum_net_ev_after_1pct": float(ev.sum()) if len(ev) else 0.0,
        "positive_ev_rate": float(ev.gt(0.0).mean()) if len(ev) else np.nan,
        "clean_exec_rate": float(pd.to_numeric(selected["clean_exec"], errors="coerce").mean()) if len(selected) else np.nan,
        "dirty_positive_rate": float(pd.to_numeric(selected["dirty_positive"], errors="coerce").mean()) if len(selected) else np.nan,
        "full_path_bad_mae_rate": float(pd.to_numeric(selected["full_path_bad_mae_1r"], errors="coerce").mean()) if len(selected) else np.nan,
        "timeout_rate": float(pd.to_numeric(selected["timeout"], errors="coerce").mean()) if len(selected) else np.nan,
    }


def _grouped_metrics(
    frame: pd.DataFrame,
    *,
    score_col: str,
    groups: list[str],
    fraction: float,
    scope: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(groups, observed=True, sort=True, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        mask = _top_fraction_mask(group[score_col], fraction)
        rows.append(
            {
                "scope": scope,
                "selector": score_col,
                **dict(zip(groups, key_values)),
                **_metrics(group, mask),
            }
        )
    return pd.DataFrame(rows)


def _stability(weekly: pd.DataFrame, selector: str) -> dict[str, Any]:
    values = pd.to_numeric(
        weekly.loc[weekly["selector"].eq(selector), "mean_net_ev_after_1pct"],
        errors="coerce",
    ).dropna().to_numpy(dtype=np.float64)
    return {
        "selector": selector,
        "weeks": int(len(values)),
        "mean_weekly_net_ev": float(np.mean(values)) if len(values) else np.nan,
        "std_weekly_net_ev": float(np.std(values, ddof=0)) if len(values) else np.nan,
        "worst_week_net_ev": float(np.min(values)) if len(values) else np.nan,
        "q01_week_net_ev": float(np.quantile(values, 0.01)) if len(values) else np.nan,
        "q10_week_net_ev": float(np.quantile(values, 0.10)) if len(values) else np.nan,
        "q33_week_net_ev": float(np.quantile(values, 1.0 / 3.0)) if len(values) else np.nan,
        "median_week_net_ev": float(np.quantile(values, 0.50)) if len(values) else np.nan,
        "q90_week_net_ev": float(np.quantile(values, 0.90)) if len(values) else np.nan,
        "positive_weeks": int(np.sum(values > 0.0)),
        "negative_weeks": int(np.sum(values <= 0.0)),
        "positive_week_rate": float(np.mean(values > 0.0)) if len(values) else np.nan,
        "weekly_ev_ac1": float(pd.Series(values).autocorr(lag=1)) if len(values) > 2 else np.nan,
    }


def _overlap(frame: pd.DataFrame, fraction: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for week, group in frame.groupby("week_start", observed=True, sort=True):
        base = _top_fraction_mask(group["score_base"], fraction)
        meta = _top_fraction_mask(group["expected_ev_rank_score"], fraction)
        base_only, meta_only = base & ~meta, meta & ~base
        ev = pd.to_numeric(group["ev_after_1pct"], errors="coerce")
        rows.append(
            {
                "week_start": week,
                "base_selected": int(base.sum()),
                "meta_selected": int(meta.sum()),
                "overlap_rows": int((base & meta).sum()),
                "jaccard": float((base & meta).sum() / max((base | meta).sum(), 1)),
                "base_only_mean_ev": float(ev[base_only].mean()) if base_only.any() else np.nan,
                "meta_only_mean_ev": float(ev[meta_only].mean()) if meta_only.any() else np.nan,
                "replacement_ev_delta": (
                    float(ev[meta_only].mean() - ev[base_only].mean())
                    if base_only.any() and meta_only.any()
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _selected_books(frame: pd.DataFrame, fraction: float) -> pd.DataFrame:
    """Return the actual weekly competing top-k books for both selectors."""
    parts: list[pd.DataFrame] = []
    for selector in SCORES.values():
        for _, group in frame.groupby("week_start", observed=True, sort=False):
            mask = _top_fraction_mask(group[selector], fraction)
            chosen = group.loc[mask].copy()
            chosen["selector"] = selector
            parts.append(chosen)
    return pd.concat(parts, ignore_index=True, copy=False)


def _selected_group_metrics(
    selected: pd.DataFrame, groups: list[str], scope: str
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in selected.groupby(["selector", *groups], observed=True, sort=True):
        values = key if isinstance(key, tuple) else (key,)
        selector, *group_values = values
        rows.append(
            {
                "scope": scope,
                "selector": selector,
                **dict(zip(groups, group_values)),
                **_metrics(group, np.ones(len(group), dtype=bool)),
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-predictions", type=Path, required=True)
    parser.add_argument("--meta-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-fraction", type=float, default=0.10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = pd.read_parquet(args.base_predictions, columns=[*KEYS, "score_base"])
    meta = pd.read_parquet(
        args.meta_predictions,
        columns=[
            *KEYS,
            *OUTCOMES,
            "hit_probability",
            "expected_ev_rank_score",
            "rank_mlp_direct",
            "policy_parent_rank",
        ],
    )
    for frame in (base, meta):
        frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
        frame["side_name"] = frame["side_name"].astype(str).str.lower()
        frame.drop_duplicates(KEYS, keep="last", inplace=True)
    frame = meta.merge(base, on=KEYS, how="inner", validate="one_to_one")
    if frame.empty:
        raise ValueError("no exact base/meta OOS row overlap")
    frame["week_start"] = frame["__ts__"].dt.floor("D") - pd.to_timedelta(
        frame["__ts__"].dt.weekday, unit="D"
    )
    frame["month"] = frame["__ts__"].dt.strftime("%Y-%m")

    weekly = pd.concat(
        [
            _grouped_metrics(frame, score_col=score, groups=["week_start"], fraction=args.top_fraction, scope="week")
            for score in SCORES.values()
        ],
        ignore_index=True,
    )
    monthly = pd.concat(
        [
            _grouped_metrics(frame, score_col=score, groups=["month"], fraction=args.top_fraction, scope="month")
            for score in SCORES.values()
        ],
        ignore_index=True,
    )
    side_archetype = pd.concat(
        [
            _grouped_metrics(frame, score_col=score, groups=["side_name", "archetype_policy_key"], fraction=args.top_fraction, scope="side_archetype")
            for score in SCORES.values()
        ],
        ignore_index=True,
    )
    stability = pd.DataFrame([_stability(weekly, score) for score in SCORES.values()])
    overlap = _overlap(frame, args.top_fraction)
    selected_books = _selected_books(frame, args.top_fraction)
    selected_side = _selected_group_metrics(selected_books, ["side_name"], "selected_side")
    selected_month_side = _selected_group_metrics(
        selected_books, ["month", "side_name"], "selected_month_side"
    )
    selected_archetype = _selected_group_metrics(
        selected_books, ["side_name", "archetype_policy_key"], "selected_side_archetype"
    )

    global_rows = []
    for score in SCORES.values():
        # Global result is the aggregation of the independently selected weekly books.
        selected = np.zeros(len(frame), dtype=bool)
        for _, group in frame.groupby("week_start", observed=True, sort=False):
            selected[group.index.to_numpy()] = _top_fraction_mask(group[score], args.top_fraction)
        global_rows.append({"selector": score, **_metrics(frame, selected)})
    global_metrics = pd.DataFrame(global_rows)
    base_global = global_metrics.loc[global_metrics["selector"].eq("score_base")].iloc[0]
    global_metrics["delta_vs_base_mean_ev"] = (
        global_metrics["mean_net_ev_after_1pct"] - float(base_global["mean_net_ev_after_1pct"])
    )

    for name, table in {
        "global_metrics.csv": global_metrics,
        "weekly_metrics.csv": weekly,
        "monthly_metrics.csv": monthly,
        "side_archetype_metrics.csv": side_archetype,
        "stability_metrics.csv": stability,
        "weekly_selection_overlap.csv": overlap,
        "selected_book_side_metrics.csv": selected_side,
        "selected_book_month_side_metrics.csv": selected_month_side,
        "selected_book_side_archetype_metrics.csv": selected_archetype,
    }.items():
        table.to_csv(args.output_dir / name, index=False)
    frame.to_parquet(args.output_dir / "base_meta_exact_overlap_rows.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "base_vs_meta_exact55_oos_metrics_v1",
        "base_predictions": str(args.base_predictions),
        "meta_predictions": str(args.meta_predictions),
        "rows": int(len(frame)),
        "start": frame["__ts__"].min().isoformat(),
        "end": frame["__ts__"].max().isoformat(),
        "top_fraction": float(args.top_fraction),
        "selection_contract": "independent top-k selection within each ISO week on identical OOS candidate rows",
        "cost_contract": "ev_after_1pct contains exactly one 1% round-trip cost",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(global_metrics.to_string(index=False))
    print(stability.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
