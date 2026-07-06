#!/usr/bin/env python3
"""Post-hoc S52 OOF score blend ablation.

This report tests whether already OOF base scores contain complementary signal:
for example, whether a pointwise precision head and a path-aware ranker head can
improve top-k executable path metrics when blended. It does not train a new
model and should be treated as a diagnostic before adding a real stacked
selector.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_gate3_side_soft_label_hpo import (  # noqa: E402
    DEFAULT_ROUND_TRIP_COST,
    LabelConfig,
    SideParams,
    _json_safe,
    _score_fold,
    _summarize_trial,
)


DEFAULT_LEDGER = Path(
    "data_perp/reports/"
    "s52_sidegeom_materialized_ranker_smoke_learnability_features_noae_20260705_v1/"
    "s52_ranker_smoke_scored_ledger.parquet"
)
DEFAULT_OUTPUT_DIR = Path(
    "data_perp/reports/"
    "s52_score_blend_ablation_learnability_features_noae_20260705_v1"
)

KEY_COLUMNS = ("month", "__ts__", "__symbol__", "side_name")
LABEL_COLUMNS = ("target_soft", "target_hard", "first_pass_good", "first_pass_bad")
METRIC_COLUMNS = (
    "u_policy_net",
    "ret_net",
    "side",
    "is_timeout",
    "mae_norm",
    "mfe_norm",
    "first_touch_net",
    "first_touch_mae_norm",
    "first_touch_mfe_norm",
    "first_touch_full_path_mae_norm",
    "mfe_1r_before_mae_1r",
    "mae_1r_before_mfe_1r",
    "max_adverse_before_mfe_1r",
    "underwater_bars_before_mfe_1r",
    "underwater_fraction_before_mfe_1r",
)


def _parse_weights(raw: str) -> list[float]:
    out: list[float] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        value = float(token)
        if value < 0.0 or value > 1.0:
            raise ValueError(f"blend weight must be in [0, 1]: {value}")
        out.append(value)
    return sorted(set(out))


def _safe_zscore(values: pd.Series, groups: pd.Series | pd.DataFrame | None) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    if groups is None:
        mean = numeric.mean()
        std = numeric.std(ddof=0)
        if not math.isfinite(float(std)) or float(std) <= 1e-12:
            return pd.Series(0.0, index=numeric.index)
        return ((numeric - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    grouped = numeric.groupby(groups, observed=True, dropna=False)
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0.0, np.nan)
    return ((numeric - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _composite_group_key(frame: pd.DataFrame, cols: list[str]) -> pd.Series:
    key = frame[cols].astype(str)
    return key.agg("\x1f".join, axis=1)


def _normalization_groups(base: pd.DataFrame, mode: str) -> pd.Series | None:
    mode = str(mode).strip().lower()
    if mode in {"none", "global"}:
        return None
    if mode == "month":
        return base["month"].astype(str)
    if mode == "month_side":
        return _composite_group_key(base, ["month", "side_name"])
    if mode == "timestamp_side":
        return _composite_group_key(base, ["month", "__ts__", "side_name"])
    raise ValueError(f"unknown normalization mode: {mode}")


def _wide_scores(ledger: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = set(KEY_COLUMNS) | {"variant", "score"}
    missing = sorted(required - set(ledger.columns))
    if missing:
        raise ValueError(f"ledger missing required columns: {missing}")

    key_frame = ledger[list(KEY_COLUMNS)].copy()
    key_frame["__ts__"] = pd.to_datetime(key_frame["__ts__"], errors="coerce")
    work = ledger.copy()
    work["__ts__"] = key_frame["__ts__"]
    dupes = work.duplicated(list(KEY_COLUMNS) + ["variant"]).sum()
    if int(dupes):
        raise ValueError(f"ledger has duplicate key/variant rows: {dupes}")

    base_cols = list(KEY_COLUMNS)
    for col in LABEL_COLUMNS + METRIC_COLUMNS:
        if col in work.columns and col not in base_cols:
            base_cols.append(col)
    base = (
        work[base_cols]
        .drop_duplicates(list(KEY_COLUMNS))
        .sort_values(list(KEY_COLUMNS))
        .reset_index(drop=True)
    )
    score_wide = (
        work.pivot(index=list(KEY_COLUMNS), columns="variant", values="score")
        .reset_index()
        .sort_values(list(KEY_COLUMNS))
        .reset_index(drop=True)
    )
    merged = base.merge(score_wide, on=list(KEY_COLUMNS), how="inner", validate="one_to_one")
    score_cols = [col for col in score_wide.columns if col not in KEY_COLUMNS]
    return merged, merged[score_cols].copy()


def _blend_scores(
    base: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    weights: list[float],
    normalization: str,
) -> dict[str, pd.Series]:
    groups = _normalization_groups(base, normalization)
    zscores = {col: _safe_zscore(scores[col], groups) for col in scores.columns}
    out: dict[str, pd.Series] = {}
    for col in scores.columns:
        out[f"single::{col}"] = zscores[col]

    variants = set(scores.columns)
    pointwise = "pointwise_lgbm"
    fullpath = "ranker_timestamp_side_fullpath_evpath"
    soft = "ranker_timestamp_side_soft_ordered_ev"

    def add_pair(name: str, left: str, right: str) -> None:
        if left not in variants or right not in variants:
            return
        for w in weights:
            out[f"{name}::w{int(round(w * 100)):03d}_{left}+{right}"] = (
                float(w) * zscores[left] + (1.0 - float(w)) * zscores[right]
            )

    add_pair("blend", pointwise, fullpath)
    add_pair("blend", pointwise, soft)
    add_pair("blend", fullpath, soft)
    if pointwise in variants and fullpath in variants and soft in variants:
        for w in weights:
            remainder = 1.0 - float(w)
            out[f"blend3::w{int(round(w * 100)):03d}_{pointwise}+path_heads"] = (
                float(w) * zscores[pointwise]
                + 0.5 * remainder * zscores[fullpath]
                + 0.5 * remainder * zscores[soft]
            )
    return out


def _evaluate_score(
    base: pd.DataFrame,
    name: str,
    score: pd.Series,
    *,
    round_trip_cost: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fold_rows: list[dict[str, Any]] = []
    eval_frame = base.copy()
    eval_frame["__score__"] = pd.to_numeric(score, errors="coerce").to_numpy(dtype=np.float64)
    for month, group in eval_frame.groupby("month", observed=True, dropna=False):
        group = group.reset_index(drop=True)
        label = group[[col for col in LABEL_COLUMNS if col in group.columns]].copy()
        positive_u = pd.to_numeric(group.get("u_policy_net", pd.Series(0.0, index=group.index)), errors="coerce").fillna(0.0).gt(0.0)
        first_bad = pd.to_numeric(label.get("first_pass_bad", pd.Series(0.0, index=group.index)), errors="coerce").fillna(0.0).gt(0.5)
        first_touch_net = pd.to_numeric(group.get("first_touch_net", pd.Series(np.nan, index=group.index)), errors="coerce")
        label["dirty_positive"] = (positive_u & first_bad).astype(np.int8)
        label["positive_u"] = positive_u.astype(np.int8)
        label["first_touch_available"] = np.isfinite(first_touch_net.to_numpy(dtype=np.float64)).astype(np.int8)
        metrics = group[[col for col in METRIC_COLUMNS if col in group.columns]].copy()
        row = _score_fold(
            pd.Series(group["__score__"], dtype=float),
            label,
            metrics,
            str(month),
            round_trip_cost=float(round_trip_cost),
        )
        row.update({"variant": name, "stage": name, "trial_number": 0, "label_name": name, "family": "score_blend"})
        fold_rows.append(row)
    neutral_side = SideParams(
        min_net_edge=0.0,
        temperature=1.0,
        mae_cap_r=0.0,
        hard_mae_cap_r=0.0,
        mae_penalty=0.0,
        mfe_min_r=0.0,
        mfe_bonus=0.0,
        mfe_mae_ratio_min=0.0,
        time_to_mfe_max_bars=0.0,
        exit_bars_min=0.0,
        exit_bars_max=0.0,
        timeout_penalty=0.0,
        late_penalty=0.0,
        dirty_positive_cap=0.0,
        timeout_cap=0.0,
        bad_mae_cap=0.0,
        post_win_mfe_min_r=0.0,
        post_win_mfe_bonus=0.0,
        first_pass_target_r=0.0,
        first_pass_bad_r=0.0,
        first_pass_reward=0.0,
        first_pass_penalty=0.0,
        adverse_pre_mfe_cap_r=0.0,
        adverse_pre_mfe_penalty=0.0,
        underwater_bars_cap=0.0,
        underwater_penalty=0.0,
        ordered_clean_floor=0.0,
        ordered_dirty_cap=0.0,
    )
    config = LabelConfig(name=name, family="score_blend", long=neutral_side, short=neutral_side)
    summary = _summarize_trial(name, 0, config, fold_rows, objective_mode="precision_topk")
    summary["variant"] = name
    summary["normalization_scope"] = ""
    return summary, fold_rows


def _write_report(output_dir: Path, summary: pd.DataFrame, folds: pd.DataFrame, manifest: dict[str, Any]) -> None:
    def fmt(df: pd.DataFrame, cols: list[str], n: int = 20) -> str:
        if df.empty:
            return "No rows."
        view = df[[col for col in cols if col in df.columns]].head(n).copy()
        for col in view.columns:
            if pd.api.types.is_float_dtype(view[col]):
                view[col] = view[col].map(lambda value: f"{float(value):.6f}" if pd.notna(value) else "")
        return view.to_markdown(index=False)

    top_cols = [
        "variant",
        "objective",
        "mean_top10_ev_weighted_first_touch_precision",
        "mean_top20_ev_weighted_first_touch_precision",
        "mean_top30_ev_weighted_first_touch_precision",
        "mean_top10_mean_first_touch_net",
        "mean_top10_first_pass_good_rate",
        "mean_top10_first_pass_bad_rate",
        "mean_top10_first_touch_full_path_bad_mae_1r_rate",
        "mean_top10_mfe_1r_before_mae_1r_rate",
        "mean_top10_mae_1r_before_mfe_1r_rate",
        "mean_top10_timeout_rate",
        "mean_long_top10_mean_first_touch_net",
        "mean_short_top10_mean_first_touch_net",
    ]
    fold_cols = [
        "variant",
        "month",
        "top10_ev_weighted_first_touch_precision",
        "top10_mean_first_touch_net",
        "top10_first_pass_good_rate",
        "top10_first_touch_full_path_bad_mae_1r_rate",
        "top10_timeout_rate",
        "long_top10_mean_first_touch_net",
        "short_top10_mean_first_touch_net",
    ]
    lines = [
        "# S52 Score Blend Ablation",
        "",
        "This is an OOF diagnostic over already-scored rows. It tests score complementarity only; it is not a leakage-safe trained stacker.",
        "",
        f"Ledger: `{manifest['ledger']}`",
        f"Rows: `{manifest['rows']}`",
        f"Normalization: `{manifest['normalization']}`",
        f"Round-trip cost: `{manifest['round_trip_cost']:.6f}`",
        "",
        "## Top Blends",
        "",
        fmt(summary, top_cols, n=30),
        "",
        "## Fold Metrics For Top 10 Variants",
        "",
        fmt(folds[folds["variant"].isin(summary["variant"].head(10))], fold_cols, n=200),
        "",
    ]
    output_dir.joinpath("s52_score_blend_ablation.md").write_text("\n".join(lines), encoding="utf-8")


def run(
    *,
    ledger_path: Path,
    output_dir: Path,
    round_trip_cost: float,
    weights: list[float],
    normalization: str,
) -> None:
    ledger = pd.read_parquet(ledger_path)
    base, score_frame = _wide_scores(ledger)
    blends = _blend_scores(base, score_frame, weights=weights, normalization=normalization)
    summaries: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for name, score in blends.items():
        summary, rows = _evaluate_score(base, name, score, round_trip_cost=float(round_trip_cost))
        summary["normalization_scope"] = str(normalization)
        summaries.append(summary)
        fold_rows.extend(rows)
    summary_df = pd.DataFrame(summaries).sort_values("objective", ascending=False).reset_index(drop=True)
    folds_df = pd.DataFrame(fold_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "s52_score_blend_summary.csv"
    folds_path = output_dir / "s52_score_blend_folds.csv"
    manifest_path = output_dir / "manifest.json"
    summary_df.to_csv(summary_path, index=False)
    folds_df.to_csv(folds_path, index=False)
    manifest = {
        "ledger": str(ledger_path),
        "output_dir": str(output_dir),
        "rows": int(len(base)),
        "variants": list(score_frame.columns),
        "blend_count": int(len(blends)),
        "weights": [float(w) for w in weights],
        "normalization": str(normalization),
        "round_trip_cost": float(round_trip_cost),
        "outputs": {
            "summary": str(summary_path),
            "folds": str(folds_path),
            "report": str(output_dir / "s52_score_blend_ablation.md"),
            "manifest": str(manifest_path),
        },
    }
    manifest_path.write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True), encoding="utf-8")
    _write_report(output_dir, summary_df, folds_df, manifest)
    print(f"wrote {summary_path}")
    print(summary_df.head(10).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--round-trip-cost", type=float, default=DEFAULT_ROUND_TRIP_COST)
    parser.add_argument("--weights", default="0,0.25,0.4,0.5,0.6,0.75,1.0")
    parser.add_argument(
        "--normalization",
        choices=("global", "month", "month_side", "timestamp_side"),
        default="month_side",
    )
    args = parser.parse_args()
    run(
        ledger_path=args.ledger,
        output_dir=args.output_dir,
        round_trip_cost=float(args.round_trip_cost),
        weights=_parse_weights(args.weights),
        normalization=str(args.normalization),
    )


if __name__ == "__main__":
    main()
