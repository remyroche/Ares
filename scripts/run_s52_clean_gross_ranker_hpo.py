#!/usr/bin/env python3
"""Focused HPO for S52 clean-gross ranker parameters.

This deliberately keeps the label geometry fixed and searches only a small set
of LightGBM ranker parameters. The goal is to test whether the current Gate 3
blocker is model capacity/regularization or label/feature learnability.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_gate3_side_soft_label_hpo import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    DEFAULT_FEATURE_LIST_CSV,
    DEFAULT_ROUND_TRIP_COST,
    _json_safe,
    _prepare_folds,
)
from scripts.run_s52_ranker_smoke import (  # noqa: E402
    DEFAULT_BEST_CONFIG,
    DEFAULT_LABELS_PATH,
    DEFAULT_MONTHS,
    _load_config,
    _parse_csv,
    _run_variant,
)


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/s52_clean_gross_ranker_hpo_20260705_v1")
EXECUTABLE_MARGIN_COST_FLOOR = 0.0100


def _finite_metric(summary: dict[str, Any], name: str, default: float = float("nan")) -> float:
    value = summary.get(name, default)
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _candidate_param_grid() -> list[dict[str, Any]]:
    return [
        {
            "name": "baseline",
            "params": {
                "n_estimators": 140,
                "learning_rate": 0.04,
                "num_leaves": 31,
                "min_child_samples": 35,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "reg_lambda": 2.0,
            },
        },
        {
            "name": "more_trees_low_lr",
            "params": {
                "n_estimators": 260,
                "learning_rate": 0.025,
                "num_leaves": 31,
                "min_child_samples": 45,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "reg_lambda": 3.0,
            },
        },
        {
            "name": "small_leaf_regularized",
            "params": {
                "n_estimators": 220,
                "learning_rate": 0.035,
                "num_leaves": 15,
                "min_child_samples": 80,
                "subsample": 0.90,
                "colsample_bytree": 0.85,
                "reg_lambda": 6.0,
            },
        },
        {
            "name": "larger_leaf_regularized",
            "params": {
                "n_estimators": 240,
                "learning_rate": 0.035,
                "num_leaves": 63,
                "min_child_samples": 80,
                "subsample": 0.80,
                "colsample_bytree": 0.75,
                "reg_lambda": 8.0,
            },
        },
        {
            "name": "shallow_high_child",
            "params": {
                "n_estimators": 300,
                "learning_rate": 0.03,
                "num_leaves": 31,
                "min_child_samples": 120,
                "subsample": 0.80,
                "colsample_bytree": 0.70,
                "reg_lambda": 10.0,
            },
        },
        {
            "name": "faster_learning_conservative",
            "params": {
                "n_estimators": 160,
                "learning_rate": 0.06,
                "num_leaves": 31,
                "min_child_samples": 80,
                "subsample": 0.75,
                "colsample_bytree": 0.85,
                "reg_lambda": 5.0,
            },
        },
        {
            "name": "deep_low_colsample",
            "params": {
                "n_estimators": 260,
                "learning_rate": 0.025,
                "num_leaves": 63,
                "min_child_samples": 45,
                "subsample": 0.75,
                "colsample_bytree": 0.70,
                "reg_lambda": 5.0,
            },
        },
        {
            "name": "compact_high_subsample",
            "params": {
                "n_estimators": 200,
                "learning_rate": 0.04,
                "num_leaves": 15,
                "min_child_samples": 45,
                "subsample": 0.95,
                "colsample_bytree": 1.00,
                "reg_lambda": 2.0,
            },
        },
        {
            "name": "longer_smooth",
            "params": {
                "n_estimators": 360,
                "learning_rate": 0.02,
                "num_leaves": 31,
                "min_child_samples": 100,
                "subsample": 0.85,
                "colsample_bytree": 0.80,
                "reg_lambda": 8.0,
            },
        },
        {
            "name": "higher_capacity_soft_reg",
            "params": {
                "n_estimators": 240,
                "learning_rate": 0.04,
                "num_leaves": 63,
                "min_child_samples": 35,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "reg_lambda": 3.0,
            },
        },
    ]


def _topk_gate_penalty(summary: dict[str, Any]) -> float:
    top10 = float(summary.get("mean_top10_ev_weighted_first_touch_precision", float("nan")))
    top20 = float(summary.get("mean_top20_ev_weighted_first_touch_precision", float("nan")))
    top30 = float(summary.get("mean_top30_ev_weighted_first_touch_precision", float("nan")))
    bad_mae = float(summary.get("mean_top10_first_touch_bad_mae_1r_rate", float("nan")))
    underwater = float(summary.get("mean_top10_mean_underwater_bars_before_mfe", float("nan")))
    mean_ev = float(summary.get("mean_top10_mean_ev", float("nan")))
    penalty = 0.0
    if math.isfinite(top10):
        penalty += max(0.0, 0.70 - top10)
    if math.isfinite(top20):
        penalty += 0.75 * max(0.0, 0.60 - top20)
    if math.isfinite(top30):
        penalty += 0.50 * max(0.0, 0.50 - top30)
    if math.isfinite(bad_mae):
        penalty += 0.50 * max(0.0, bad_mae - 0.15)
    if math.isfinite(underwater):
        penalty += 0.03 * max(0.0, underwater - 10.0)
    if math.isfinite(mean_ev):
        penalty += 5.0 * max(0.0, -mean_ev)
    return float(penalty)


def _s52_topk_path_objective(summary: dict[str, Any]) -> float:
    """Primary S52 ranker-HPO objective.

    S52 should be selected by the tradable cross-sectional slice: EV-weighted
    clean first-touch precision at top-k, with explicit path-order penalties.
    Net at the materialized cost remains a diagnostic, but gross first-touch
    edge is included so high-cost materializations do not erase learnability.
    """

    top10 = _finite_metric(summary, "mean_top10_ev_weighted_first_touch_precision", 0.0)
    top20 = _finite_metric(summary, "mean_top20_ev_weighted_first_touch_precision", 0.0)
    top30 = _finite_metric(summary, "mean_top30_ev_weighted_first_touch_precision", 0.0)
    long_top10 = _finite_metric(summary, "mean_long_top10_ev_weighted_first_touch_precision", float("nan"))
    short_top10 = _finite_metric(summary, "mean_short_top10_ev_weighted_first_touch_precision", float("nan"))
    side_min = min(v for v in (long_top10, short_top10) if math.isfinite(v)) if (
        math.isfinite(long_top10) or math.isfinite(short_top10)
    ) else 0.0
    side_gap = abs(long_top10 - short_top10) if math.isfinite(long_top10) and math.isfinite(short_top10) else 0.0

    gross10 = _finite_metric(summary, "mean_top10_mean_first_touch_gross", 0.0)
    net10 = _finite_metric(summary, "mean_top10_mean_first_touch_net", 0.0)
    ev10 = _finite_metric(summary, "mean_top10_mean_ev", 0.0)
    bad_mae = _finite_metric(summary, "mean_top10_first_touch_bad_mae_1r_rate", 1.0)
    mae_before = _finite_metric(summary, "mean_top10_mae_1r_before_mfe_1r_rate", 1.0)
    mfe_before = _finite_metric(summary, "mean_top10_mfe_1r_before_mae_1r_rate", 0.0)
    timeout = _finite_metric(summary, "mean_top10_timeout_rate", 1.0)
    underwater = _finite_metric(summary, "mean_top10_mean_underwater_bars_before_mfe", 20.0)
    underwater_frac = _finite_metric(summary, "mean_top10_mean_underwater_fraction_before_mfe", 1.0)
    max_adverse = _finite_metric(summary, "mean_top10_mean_max_adverse_before_mfe_1r", 2.0)

    objective = (
        1.60 * top10
        + 1.05 * top20
        + 0.70 * top30
        + 0.45 * side_min
        + 0.40 * mfe_before
        + 20.0 * gross10
        + 6.0 * max(net10, -0.01)
        + 3.0 * max(ev10, -0.01)
        - 0.55 * bad_mae
        - 0.60 * mae_before
        - 0.20 * timeout
        - 0.18 * max(underwater - 6.0, 0.0)
        - 0.35 * max(underwater_frac - 0.35, 0.0)
        - 0.22 * max(max_adverse - 1.0, 0.0)
        - 0.20 * side_gap
    )
    if bad_mae > 0.25:
        objective -= 0.50 * (bad_mae - 0.25)
    if mae_before > 0.35:
        objective -= 0.45 * (mae_before - 0.35)
    if top10 < 0.70:
        objective -= 0.80 * (0.70 - top10)
    if top20 < 0.55:
        objective -= 0.45 * (0.55 - top20)
    return float(objective) if math.isfinite(objective) else float("-inf")


def _metric_or_derived_margin(summary: dict[str, Any], tag: str) -> float:
    explicit = _finite_metric(summary, f"mean_{tag}_mean_first_touch_executable_margin", float("nan"))
    if math.isfinite(explicit):
        return explicit
    gross = _finite_metric(summary, f"mean_{tag}_mean_first_touch_gross", float("nan"))
    net = _finite_metric(summary, f"mean_{tag}_mean_first_touch_net", float("nan"))
    if not math.isfinite(gross):
        return 0.0
    implied_cost = gross - net if math.isfinite(net) else float(EXECUTABLE_MARGIN_COST_FLOOR)
    return float(gross - max(float(EXECUTABLE_MARGIN_COST_FLOOR), implied_cost))


def _s52_executable_margin_topk_objective(summary: dict[str, Any]) -> float:
    """Top-k objective that rewards clean precision only when it clears cost."""

    base = _s52_topk_path_objective(summary)
    margin10 = _metric_or_derived_margin(summary, "top10")
    margin20 = _metric_or_derived_margin(summary, "top20")
    margin30 = _metric_or_derived_margin(summary, "top30")
    gross_minus_1pct10 = _finite_metric(summary, "mean_top10_mean_first_touch_gross_minus_1pct", margin10)
    hit_margin10 = _finite_metric(summary, "mean_top10_hit_first_touch_executable_margin", 0.0)
    hit_margin20 = _finite_metric(summary, "mean_top20_hit_first_touch_executable_margin", 0.0)
    long_margin = _finite_metric(summary, "mean_long_top10_mean_first_touch_executable_margin", float("nan"))
    short_margin = _finite_metric(summary, "mean_short_top10_mean_first_touch_executable_margin", float("nan"))
    side_margin_min = min(v for v in (long_margin, short_margin) if math.isfinite(v)) if (
        math.isfinite(long_margin) or math.isfinite(short_margin)
    ) else 0.0
    side_margin_gap = abs(long_margin - short_margin) if math.isfinite(long_margin) and math.isfinite(short_margin) else 0.0

    objective = (
        base
        + 28.0 * margin10
        + 16.0 * margin20
        + 8.0 * margin30
        + 8.0 * gross_minus_1pct10
        + 0.45 * hit_margin10
        + 0.25 * hit_margin20
        + 10.0 * side_margin_min
        - 8.0 * max(0.0, -margin10)
        - 5.0 * max(0.0, -margin20)
        - 3.0 * max(0.0, -side_margin_min)
        - 2.0 * side_margin_gap
    )
    if margin10 <= 0.0:
        objective -= 0.60
    if hit_margin10 < 0.50:
        objective -= 0.35 * (0.50 - hit_margin10)
    return float(objective) if math.isfinite(objective) else float("-inf")


def _select_hpo_objective(summary: dict[str, Any], mode: str) -> float:
    mode_norm = str(mode or "s52_executable_margin_topk").strip().lower()
    if mode_norm in {"s52_executable_margin_topk", "executable_margin_topk", "exec_margin_topk"}:
        return _s52_executable_margin_topk_objective(summary)
    if mode_norm in {"s52_topk_path", "topk_path", "ev_weighted_topk_path"}:
        return _s52_topk_path_objective(summary)
    if mode_norm in {"objective", "inherited", "inherited_objective"}:
        return _finite_metric(summary, "objective", float("-inf"))
    raise ValueError(f"unknown selection objective: {mode}")


def run_hpo(
    *,
    labels_path: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    best_config_path: Path,
    output_dir: Path,
    months: list[str],
    variants: list[str],
    sample_weight_modes: list[str],
    max_train_rows: int,
    max_trials: int,
    round_trip_cost: float,
    seed: int,
    include_ae_gmm_state_features: bool,
    ae_gmm_state_feature_max_train_rows: int,
    ae_gmm_state_feature_max_iter: int,
    ae_gmm_state_feature_seed: int,
    target_source: str,
    selection_objective: str,
    ae_gmm_fold_cache_dir: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _load_config(best_config_path)
    folds, fold_manifest = _prepare_folds(
        labels_path=labels_path,
        feature_dir=feature_dir,
        feature_list_csv=feature_list_csv,
        months=months,
        spread_baseline_path=None,
        spread_rank_column="p75_spread_bps",
        target_symbol_count=None,
        max_feature_store_features=None,
        include_ae_gmm_state_features=include_ae_gmm_state_features,
        ae_gmm_state_feature_max_train_rows=int(ae_gmm_state_feature_max_train_rows),
        ae_gmm_state_feature_max_iter=int(ae_gmm_state_feature_max_iter),
        seed=int(seed),
        ae_gmm_state_feature_seed=int(ae_gmm_state_feature_seed),
        ae_gmm_fold_cache_dir=ae_gmm_fold_cache_dir,
    )
    candidates = _candidate_param_grid()[: max(1, int(max_trials))]
    summaries: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for trial_i, candidate in enumerate(candidates):
        params = dict(candidate["params"])
        for variant in variants:
            for weight_mode in sample_weight_modes:
                summary, rows, _ledgers = _run_variant(
                    variant=variant,
                    folds=folds,
                    config=config,
                    max_train_rows=int(max_train_rows),
                    round_trip_cost=float(round_trip_cost),
                    path_order_mode="s52_first_touch",
                    target_utility_mode="geometry_only",
                    target_source=str(target_source),
                    seed=int(seed) + 1000 * trial_i,
                    ranker_params=params,
                    sample_weight_mode=str(weight_mode),
                )
                summary.update(
                    {
                        "trial_number": int(trial_i),
                        "trial_name": str(candidate["name"]),
                        "variant": str(variant),
                        "sample_weight_mode": str(weight_mode),
                        "ranker_params": json.dumps(_json_safe(params), sort_keys=True),
                        "gate_shortfall_penalty": _topk_gate_penalty(summary),
                        "hpo_selection_objective": _select_hpo_objective(summary, selection_objective),
                        "hpo_selection_objective_mode": str(selection_objective),
                    }
                )
                fold_rows.extend(
                    {
                        **row,
                        "trial_number": int(trial_i),
                        "trial_name": str(candidate["name"]),
                        "sample_weight_mode": str(weight_mode),
                        "ranker_params": json.dumps(_json_safe(params), sort_keys=True),
                    }
                    for row in rows
                )
                summaries.append(summary)
    summary_df = pd.DataFrame(summaries).sort_values(
        ["hpo_selection_objective", "gate_shortfall_penalty", "objective"],
        ascending=[False, True, False],
    ).reset_index(drop=True)
    fold_df = pd.DataFrame(fold_rows)
    paths = {
        "summary": output_dir / "s52_clean_gross_ranker_hpo_summary.csv",
        "folds": output_dir / "s52_clean_gross_ranker_hpo_folds.csv",
        "manifest": output_dir / "manifest.json",
        "report": output_dir / "s52_clean_gross_ranker_hpo.md",
    }
    summary_df.to_csv(paths["summary"], index=False)
    fold_df.to_csv(paths["folds"], index=False)
    manifest = {
        **fold_manifest,
        "scope": "s52_clean_gross_ranker_hpo",
        "labels_path": str(labels_path),
        "feature_dir": str(feature_dir),
        "feature_list_csv": str(feature_list_csv),
        "best_config_path": str(best_config_path),
        "output_dir": str(output_dir),
        "months": [str(m) for m in months],
        "variants": [str(v) for v in variants],
        "sample_weight_modes": [str(v) for v in sample_weight_modes],
        "max_train_rows": int(max_train_rows),
        "max_trials": int(max_trials),
        "round_trip_cost": float(round_trip_cost),
        "seed": int(seed),
        "model_seed": int(seed),
        "ae_gmm_state_feature_seed": int(ae_gmm_state_feature_seed),
        "target_source": str(target_source),
        "objective_mode": "precision_topk",
        "hpo_selection_objective_mode": str(selection_objective),
        "ae_gmm_fold_cache_dir": str(ae_gmm_fold_cache_dir) if ae_gmm_fold_cache_dir is not None else None,
        "config": asdict(config),
        "candidates": candidates,
        "outputs": {k: str(v) for k, v in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    _write_report(paths["report"], summary_df, fold_df, manifest)
    return {
        "output_dir": str(output_dir),
        "summary": str(paths["summary"]),
        "folds": str(paths["folds"]),
        "report": str(paths["report"]),
        "top": _json_safe(summary_df.head(5).to_dict(orient="records")),
    }


def _write_report(path: Path, summary_df: pd.DataFrame, fold_df: pd.DataFrame, manifest: dict[str, Any]) -> None:
    def fmt(df: pd.DataFrame, cols: list[str]) -> str:
        if df.empty:
            return "No rows."
        view = df[[c for c in cols if c in df.columns]].copy()
        for col in view.columns:
            if pd.api.types.is_float_dtype(view[col]):
                view[col] = view[col].map(lambda v: f"{float(v):.4f}" if pd.notna(v) else "")
        return view.to_markdown(index=False)

    cols = [
        "trial_number",
        "trial_name",
        "variant",
        "sample_weight_mode",
        "objective",
        "hpo_selection_objective",
        "gate_shortfall_penalty",
        "mean_top10_ev_weighted_first_touch_precision",
        "mean_top20_ev_weighted_first_touch_precision",
        "mean_top30_ev_weighted_first_touch_precision",
        "mean_top10_first_touch_bad_mae_1r_rate",
        "mean_top10_mean_underwater_bars_before_mfe",
        "mean_top10_mean_ev",
        "mean_top10_mean_first_touch_gross_minus_1pct",
        "mean_top10_mean_first_touch_executable_margin",
        "mean_top10_hit_first_touch_executable_margin",
        "mean_long_top10_ev_weighted_first_touch_precision",
        "mean_short_top10_ev_weighted_first_touch_precision",
        "mean_long_top10_mean_first_touch_executable_margin",
        "mean_short_top10_mean_first_touch_executable_margin",
    ]
    fold_cols = [
        "trial_name",
        "variant",
        "sample_weight_mode",
        "month",
        "top10_ev_weighted_first_touch_precision",
        "top20_ev_weighted_first_touch_precision",
        "top30_ev_weighted_first_touch_precision",
        "top10_first_touch_bad_mae_1r_rate",
        "top10_mean_underwater_bars_before_mfe",
        "top10_mean_ev",
        "top10_mean_first_touch_gross_minus_1pct",
        "top10_mean_first_touch_executable_margin",
        "top10_hit_first_touch_executable_margin",
    ]
    lines = [
        "# S52 Clean-Gross Ranker HPO",
        "",
        "Fixed label geometry; bounded LightGBM ranker parameter search.",
        "",
        f"Rows: `{manifest.get('rows')}`",
        f"Symbols: `{manifest.get('symbols')}`",
        f"Months: `{', '.join(manifest.get('fold_months', manifest.get('months', [])))}`",
        f"Variants: `{', '.join(manifest.get('variants', []))}`",
        "",
        "## Best Trials",
        "",
        fmt(summary_df.head(20), cols),
        "",
        "## Best Trial Folds",
        "",
    ]
    best = summary_df.iloc[0] if len(summary_df) else None
    if best is not None:
        mask = (
            fold_df["trial_name"].astype(str).eq(str(best["trial_name"]))
            & fold_df["variant"].astype(str).eq(str(best["variant"]))
            & fold_df["sample_weight_mode"].astype(str).eq(str(best["sample_weight_mode"]))
        )
        lines.append(fmt(fold_df[mask], fold_cols))
    else:
        lines.append("No rows.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--best-config-path", type=Path, default=DEFAULT_BEST_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--months", default=",".join(DEFAULT_MONTHS))
    parser.add_argument(
        "--variants",
        default="ranker_timestamp_cleangross,ranker_timestamp_side_cleangross",
        help="Comma-separated ranker variants to test.",
    )
    parser.add_argument(
        "--sample-weight-modes",
        default="base",
        help="Comma-separated sample weight modes: base,execres_clean_dirty,long_clean_dirty.",
    )
    parser.add_argument("--max-train-rows", type=int, default=150_000)
    parser.add_argument("--max-trials", type=int, default=10)
    parser.add_argument("--round-trip-cost", type=float, default=DEFAULT_ROUND_TRIP_COST)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-ae-gmm-state-features", action="store_true")
    parser.add_argument("--ae-gmm-state-feature-max-train-rows", type=int, default=60_000)
    parser.add_argument("--ae-gmm-state-feature-max-iter", type=int, default=32)
    parser.add_argument("--ae-gmm-state-feature-seed", type=int, default=42)
    parser.add_argument(
        "--ae-gmm-fold-cache-dir",
        type=Path,
        default=None,
        help="Optional cache for fold-level AE/GMM-augmented train/valid matrices.",
    )
    parser.add_argument(
        "--target-source",
        choices=("hpo_config", "materialized"),
        default="materialized",
    )
    parser.add_argument(
        "--selection-objective",
        choices=("s52_executable_margin_topk", "s52_topk_path", "inherited_objective"),
        default="s52_executable_margin_topk",
        help="Primary HPO sort key. s52_executable_margin_topk adds a 1%% cost-floor executable-margin objective.",
    )
    args = parser.parse_args()
    result = run_hpo(
        labels_path=args.labels_path,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        best_config_path=args.best_config_path,
        output_dir=args.output_dir,
        months=_parse_csv(args.months, DEFAULT_MONTHS),
        variants=_parse_csv(args.variants, ()),
        sample_weight_modes=_parse_csv(args.sample_weight_modes, ("base",)),
        max_train_rows=int(args.max_train_rows),
        max_trials=int(args.max_trials),
        round_trip_cost=float(args.round_trip_cost),
        seed=int(args.seed),
        include_ae_gmm_state_features=not bool(args.no_ae_gmm_state_features),
        ae_gmm_state_feature_max_train_rows=int(args.ae_gmm_state_feature_max_train_rows),
        ae_gmm_state_feature_max_iter=int(args.ae_gmm_state_feature_max_iter),
        ae_gmm_state_feature_seed=int(args.ae_gmm_state_feature_seed),
        target_source=str(args.target_source),
        selection_objective=str(args.selection_objective),
        ae_gmm_fold_cache_dir=args.ae_gmm_fold_cache_dir,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
