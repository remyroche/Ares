#!/usr/bin/env python3
"""Learned baseline-preserving admission overlay for guarded execution.

This diagnostic keeps the current anchored-adaptive accepted trades fixed and
learns whether broader-source incremental accepted rows can be admitted using
only prior-fold outcomes and pre-entry features. It is not a promotion artifact;
passing rows still need materialized frozen replay and breadth review.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.simple_policy_optimiser import _json_safe  # noqa: E402
from scripts.report_guarded_execution_additive_overlay import (  # noqa: E402
    BASELINE_DIR,
    DEFAULT_VARIANTS,
    REPORT_ROOT,
    _candidate_incremental,
    _fmt_table,
    _load_variant_accepted,
    _portfolio_capacity_filter,
    _score_rows,
)
from scripts.validate_meta_handoff_execution_guard_walkforward import _fit_predict_scores  # noqa: E402


DEFAULT_OUT_DIR = REPORT_ROOT / "guarded_execution_learned_overlay_20260703_v1"
FEATURE_COLUMNS = (
    "normalized_rank_score",
    "effective_rank_score",
    "adaptive_guard_score_oof",
    "adaptive_guard_margin",
    "adaptive_guard_keep_frac",
    "rank_minus_joint_bad",
    "rank_minus_joint_timeout",
    "oof_regime_centroid_similarity_train",
    "archetype_meta_bad_risk",
    "archetype_meta_timeout_risk",
    "archetype_joint_bad_risk",
    "archetype_joint_timeout_risk",
    "neg_archetype_meta_bad_risk",
    "neg_archetype_meta_timeout_risk",
    "neg_archetype_joint_bad_risk",
    "neg_archetype_joint_timeout_risk",
    "effective_rank_minus_joint_bad",
    "effective_rank_minus_meta_bad",
    "effective_rank_minus_joint_bad_timeout",
    "guard_score_minus_joint_bad",
    "guard_margin_minus_joint_bad",
    "calibrated_minus_joint_bad",
    "neg_joint_bad_timeout_sum",
    "horizon_hours",
    "barrier_multiplier",
    "policy_effective_barrier_pct",
    "side_code",
    "is_short",
)
METHODS = (
    "risk_composite_rule",
    "exec_net_regressor",
    "clean_exit_classifier",
    "bad_exit_veto",
    "exec_clean_bad_blend",
    "exec_bad_blend",
)


def _prepare_overlay_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    side = out.get("side", pd.Series("", index=out.index)).astype(str).str.lower()
    out["side_code"] = np.where(side.str.contains("short"), -1.0, 1.0).astype("float32")
    out["is_short"] = (out["side_code"] < 0.0).astype("float32")
    out["net_return"] = pd.to_numeric(out.get("accepted_net_return", out.get("net_return", 0.0)), errors="coerce").fillna(0.0)
    out["clean_executable"] = out["net_return"].gt(0.0) & ~out["is_full_sl"].astype(bool) & ~out["is_timeout"].astype(bool)
    out["bad_executable"] = out["is_full_sl"].astype(bool) | out["is_timeout"].astype(bool) | out["net_return"].lt(0.0)
    for col in FEATURE_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _features(frame: pd.DataFrame) -> list[str]:
    return [col for col in FEATURE_COLUMNS if col in frame.columns]


def _standardize(train_scores: np.ndarray, eval_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train = np.asarray(train_scores, dtype=np.float64)
    eval_arr = np.asarray(eval_scores, dtype=np.float64)
    finite = train[np.isfinite(train)]
    if finite.size == 0:
        return np.zeros_like(train, dtype=np.float32), np.zeros_like(eval_arr, dtype=np.float32)
    mean = float(np.mean(finite))
    std = float(np.std(finite))
    if not np.isfinite(std) or std < 1.0e-9:
        std = 1.0
    return ((train - mean) / std).astype(np.float32), ((eval_arr - mean) / std).astype(np.float32)


def _predict_scores(
    *,
    method: str,
    train: pd.DataFrame,
    eval_frame: pd.DataFrame,
    features: list[str],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    if method in {"risk_composite_rule", "exec_net_regressor", "clean_exit_classifier", "bad_exit_veto"}:
        return _fit_predict_scores(method, train, eval_frame, features, seed=int(seed))
    exec_train, exec_eval, exec_name = _fit_predict_scores("exec_net_regressor", train, eval_frame, features, seed=int(seed))
    clean_train, clean_eval, clean_name = _fit_predict_scores("clean_exit_classifier", train, eval_frame, features, seed=int(seed) + 17)
    bad_train, bad_eval, bad_name = _fit_predict_scores("bad_exit_veto", train, eval_frame, features, seed=int(seed) + 31)
    exec_train_z, exec_eval_z = _standardize(exec_train, exec_eval)
    clean_train_z, clean_eval_z = _standardize(clean_train, clean_eval)
    bad_train_z, bad_eval_z = _standardize(bad_train, bad_eval)
    if method == "exec_bad_blend":
        return (
            (0.70 * exec_train_z + 0.30 * bad_train_z).astype(np.float32),
            (0.70 * exec_eval_z + 0.30 * bad_eval_z).astype(np.float32),
            f"blend({exec_name},{bad_name})",
        )
    return (
        (0.50 * exec_train_z + 0.25 * clean_train_z + 0.25 * bad_train_z).astype(np.float32),
        (0.50 * exec_eval_z + 0.25 * clean_eval_z + 0.25 * bad_eval_z).astype(np.float32),
        f"blend({exec_name},{clean_name},{bad_name})",
    )


def _threshold_grid(values: pd.Series) -> list[float]:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if numeric.empty:
        return []
    qs = [0.0, 0.10, 0.25, 0.35, 0.50, 0.65, 0.75, 0.85, 0.90]
    return sorted({float(numeric.quantile(q)) for q in qs})


def _objective(stats: dict[str, float], *, min_trades: int) -> float:
    trades = int(stats["trades"])
    if trades < int(min_trades):
        return -1.0e9 + float(trades)
    return float(
        0.00002 * stats["net_pnl"]
        + (0.0 if np.isnan(stats["mean_return"]) else stats["mean_return"])
        - 0.08 * (0.0 if np.isnan(stats["full_sl_rate"]) else stats["full_sl_rate"])
        - 0.03 * (0.0 if np.isnan(stats["timeout_rate"]) else stats["timeout_rate"])
    )


def _select_threshold(train: pd.DataFrame, *, min_train_addons: int) -> tuple[float, float, dict[str, float]]:
    best: tuple[float, float, dict[str, float]] = (float("inf"), 0.0, _score_rows(pd.DataFrame()))
    if train.empty or "overlay_model_score" not in train.columns:
        return best
    for threshold in _threshold_grid(train["overlay_model_score"]):
        kept = train.loc[pd.to_numeric(train["overlay_model_score"], errors="coerce").ge(float(threshold))]
        stats = _score_rows(kept)
        score = _objective(stats, min_trades=int(min_train_addons))
        if score > best[1]:
            best = (float(threshold), float(score), stats)
    if best[1] <= 0.0:
        return float("inf"), float(best[1]), _score_rows(pd.DataFrame())
    return best


def _run_variant_method(
    *,
    baseline: pd.DataFrame,
    incremental: pd.DataFrame,
    variant_name: str,
    method: str,
    min_train_addons: int,
    max_open: int,
    max_open_per_side: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = _features(incremental)
    weeks = sorted(baseline["validation_week"].dropna().astype(str).unique())
    rows: list[dict[str, Any]] = []
    kept_frames: list[pd.DataFrame] = []
    for fold_id, week in enumerate(weeks):
        base_eval = baseline.loc[baseline["validation_week"].astype(str).eq(week)].copy()
        train = incremental.loc[incremental["validation_week"].astype(str) < str(week)].copy()
        eval_pool = incremental.loc[incremental["validation_week"].astype(str).eq(week)].copy()
        model_name = "no_model"
        threshold = float("inf")
        train_score = 0.0
        train_stats = _score_rows(pd.DataFrame())
        if len(train) >= int(min_train_addons) and not eval_pool.empty:
            train_scores, eval_scores, model_name = _predict_scores(
                method=method,
                train=train,
                eval_frame=eval_pool,
                features=features,
                seed=int(seed) + int(fold_id),
            )
            train = train.copy()
            eval_pool = eval_pool.copy()
            train["overlay_model_score"] = train_scores
            eval_pool["overlay_model_score"] = eval_scores
            threshold, train_score, train_stats = _select_threshold(train, min_train_addons=int(min_train_addons))
        if not np.isfinite(float(threshold)):
            pre_capacity = eval_pool.iloc[0:0].copy()
        else:
            scores = pd.to_numeric(eval_pool.get("overlay_model_score"), errors="coerce").replace([np.inf, -np.inf], np.nan)
            pre_capacity = eval_pool.loc[scores.ge(float(threshold)).fillna(False)].copy()
        add_eval, rejected_capacity = _portfolio_capacity_filter(
            baseline=baseline.loc[baseline["validation_week"].astype(str).le(str(week))].copy(),
            add_ons=pre_capacity,
            max_open=int(max_open),
            max_open_per_side=int(max_open_per_side),
        )
        if not add_eval.empty:
            add_eval["overlay_method"] = method
            add_eval["overlay_model_name"] = model_name
            add_eval["overlay_threshold"] = float(threshold)
            add_eval["overlay_train_score"] = float(train_score)
            add_eval["variant_name"] = variant_name
            kept_frames.append(add_eval)
        combined = pd.concat([base_eval, add_eval], ignore_index=True) if not add_eval.empty else base_eval
        base_stats = _score_rows(base_eval)
        add_stats = _score_rows(add_eval)
        combined_stats = _score_rows(combined)
        rows.append(
            {
                "fold_id": int(fold_id),
                "validation_week": week,
                "variant_name": variant_name,
                "method": method,
                "model_name": model_name,
                "feature_count": int(len(features)),
                "threshold": float(threshold),
                "train_selector_score": float(train_score),
                "train_addon_trades": int(train_stats["trades"]),
                "train_addon_net_pnl": float(train_stats["net_pnl"]),
                "baseline_trades": int(base_stats["trades"]),
                "baseline_net_pnl": float(base_stats["net_pnl"]),
                "addon_pool_trades": int(len(eval_pool)),
                "addon_pre_capacity_trades": int(len(pre_capacity)),
                "addon_trades": int(add_stats["trades"]),
                "addon_net_pnl": float(add_stats["net_pnl"]),
                "addon_full_sl_rate": float(add_stats["full_sl_rate"]) if not np.isnan(add_stats["full_sl_rate"]) else np.nan,
                "addon_timeout_rate": float(add_stats["timeout_rate"]) if not np.isnan(add_stats["timeout_rate"]) else np.nan,
                "capacity_rejections": int(len(rejected_capacity)),
                "combined_trades": int(combined_stats["trades"]),
                "combined_net_pnl": float(combined_stats["net_pnl"]),
                "combined_full_sl_rate": float(combined_stats["full_sl_rate"]) if not np.isnan(combined_stats["full_sl_rate"]) else np.nan,
                "combined_timeout_rate": float(combined_stats["timeout_rate"]) if not np.isnan(combined_stats["timeout_rate"]) else np.nan,
                "combined_hit_rate": float(combined_stats["hit_rate"]) if not np.isnan(combined_stats["hit_rate"]) else np.nan,
            }
        )
    kept = pd.concat(kept_frames, ignore_index=True) if kept_frames else pd.DataFrame()
    return pd.DataFrame(rows), kept


def _summarise_folds(folds: pd.DataFrame, *, variant: str, method: str) -> dict[str, Any]:
    if folds.empty:
        return {"variant_name": variant, "method": method, "overlay_gate_pass": False}
    accepted = pd.to_numeric(folds["combined_trades"], errors="coerce").fillna(0)
    net = pd.to_numeric(folds["combined_net_pnl"], errors="coerce").fillna(0.0)
    total = float(max(accepted.sum(), 1.0))
    full_sl = pd.to_numeric(folds["combined_full_sl_rate"], errors="coerce").fillna(0.0)
    timeout = pd.to_numeric(folds["combined_timeout_rate"], errors="coerce").fillna(0.0)
    addon_trades = pd.to_numeric(folds["addon_trades"], errors="coerce").fillna(0)
    addon_net = pd.to_numeric(folds["addon_net_pnl"], errors="coerce").fillna(0.0)
    positive_folds = int(net.gt(0.0).sum())
    rows = int(len(folds))
    weighted_full_sl = float((full_sl * accepted).sum() / total)
    weighted_timeout = float((timeout * accepted).sum() / total)
    pass_gate = bool(
        net.sum() > 2570.3172320144395
        and positive_folds / max(rows, 1) >= 5 / 6
        and weighted_full_sl <= 0.166667
        and weighted_timeout <= 0.50
        and addon_trades.sum() > 0
    )
    return {
        "variant_name": variant,
        "method": method,
        "overlay_gate_pass": pass_gate,
        "folds": rows,
        "combined_net_pnl": float(net.sum()),
        "combined_positive_folds": positive_folds,
        "combined_positive_fold_share": float(positive_folds / max(rows, 1)),
        "combined_trades": int(accepted.sum()),
        "baseline_trades": int(pd.to_numeric(folds["baseline_trades"], errors="coerce").fillna(0).sum()),
        "addon_trades": int(addon_trades.sum()),
        "addon_net_pnl": float(addon_net.sum()),
        "weighted_full_sl_rate": weighted_full_sl,
        "weighted_timeout_rate": weighted_timeout,
        "mean_addon_trades": float(addon_trades.mean()),
        "min_addon_trades": int(addon_trades.min()) if len(addon_trades) else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=BASELINE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-train-addons", type=int, default=8)
    parser.add_argument("--max-open", type=int, default=8)
    parser.add_argument("--max-open-per-side", type=int, default=4)
    parser.add_argument("--seed", type=int, default=739391)
    args = parser.parse_args()

    baseline = _prepare_overlay_frame(_load_variant_accepted(args.baseline_dir))
    if baseline.empty:
        raise ValueError(f"Missing baseline accepted decisions under {args.baseline_dir}")

    summary_rows: list[dict[str, Any]] = []
    fold_frames: list[pd.DataFrame] = []
    kept_frames: list[pd.DataFrame] = []
    incremental_rows: list[dict[str, Any]] = []
    for variant_name, variant_dir in DEFAULT_VARIANTS:
        variant = _prepare_overlay_frame(_load_variant_accepted(variant_dir))
        if variant.empty:
            continue
        incremental = _prepare_overlay_frame(_candidate_incremental(variant, baseline))
        incremental_rows.append(
            {
                "variant_name": variant_name,
                "variant_accepted_trades": int(len(variant)),
                "incremental_pool_trades": int(len(incremental)),
                "incremental_pool_net_pnl": float(incremental["accepted_net_pnl"].sum()) if not incremental.empty else 0.0,
                "incremental_pool_full_sl_rate": float(incremental["is_full_sl"].mean()) if not incremental.empty else np.nan,
                "incremental_pool_timeout_rate": float(incremental["is_timeout"].mean()) if not incremental.empty else np.nan,
            }
        )
        if incremental.empty:
            continue
        for method in METHODS:
            folds, kept = _run_variant_method(
                baseline=baseline,
                incremental=incremental,
                variant_name=variant_name,
                method=method,
                min_train_addons=int(args.min_train_addons),
                max_open=int(args.max_open),
                max_open_per_side=int(args.max_open_per_side),
                seed=int(args.seed),
            )
            if not folds.empty:
                fold_frames.append(folds)
            if not kept.empty:
                kept_frames.append(kept)
            summary_rows.append(_summarise_folds(folds, variant=variant_name, method=method))

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values(
            ["overlay_gate_pass", "combined_net_pnl", "addon_trades", "weighted_full_sl_rate"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
    folds_all = pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()
    kept_all = pd.concat(kept_frames, ignore_index=True) if kept_frames else pd.DataFrame()
    incremental = pd.DataFrame(incremental_rows)
    pass_count = int(summary.get("overlay_gate_pass", pd.Series(dtype=bool)).astype(bool).sum()) if not summary.empty else 0
    best = summary.iloc[0].to_dict() if not summary.empty else {}
    conclusion = {
        "status": "learned_overlay_candidate_found" if pass_count else "no_learned_overlay_candidate",
        "overlay_gate_pass_count": pass_count,
        "best_variant": best.get("variant_name", ""),
        "best_method": best.get("method", ""),
        "best_combined_trades": int(best.get("combined_trades", 0) or 0),
        "best_addon_trades": int(best.get("addon_trades", 0) or 0),
        "best_combined_net_pnl": float(best.get("combined_net_pnl", np.nan)) if best else np.nan,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": args.out_dir / "guarded_execution_learned_overlay_summary.csv",
        "folds": args.out_dir / "guarded_execution_learned_overlay_folds.csv",
        "incremental_pool": args.out_dir / "guarded_execution_learned_overlay_incremental_pool.csv",
        "kept_addons": args.out_dir / "guarded_execution_learned_overlay_kept_addons.parquet",
        "manifest": args.out_dir / "guarded_execution_learned_overlay_manifest.json",
        "report": args.out_dir / "guarded_execution_learned_overlay_report.md",
    }
    summary.to_csv(paths["summary"], index=False)
    folds_all.to_csv(paths["folds"], index=False)
    incremental.to_csv(paths["incremental_pool"], index=False)
    kept_all.to_parquet(paths["kept_addons"], index=False)
    manifest = {
        "generated_by": "report_guarded_execution_learned_overlay",
        "out_dir": str(args.out_dir),
        "baseline_dir": str(args.baseline_dir),
        "variants": [{"name": name, "directory": str(path)} for name, path in DEFAULT_VARIANTS],
        "methods": list(METHODS),
        "feature_columns": list(FEATURE_COLUMNS),
        "min_train_addons": int(args.min_train_addons),
        "max_open": int(args.max_open),
        "max_open_per_side": int(args.max_open_per_side),
        "diagnostic_only": True,
        "diagnostic_caveat": "Baseline trades are preserved and add-ons are learned from prior-fold incremental outcomes; this is not a full frozen replay.",
        "conclusion": conclusion,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    report_lines = [
        "# Guarded Execution Learned Overlay Diagnostic",
        "",
        f"Status: `{conclusion['status']}`",
        "",
        "This diagnostic preserves the current 36-trade anchored-adaptive baseline and learns add-on admission from prior-fold incremental rows only.",
        "",
        "## Summary",
        "",
        _fmt_table(
            summary,
            [
                "variant_name",
                "method",
                "overlay_gate_pass",
                "combined_net_pnl",
                "combined_positive_fold_share",
                "combined_trades",
                "addon_trades",
                "addon_net_pnl",
                "weighted_full_sl_rate",
                "weighted_timeout_rate",
            ],
            max_rows=80,
        ),
        "",
        "## Incremental Pools",
        "",
        _fmt_table(
            incremental,
            [
                "variant_name",
                "variant_accepted_trades",
                "incremental_pool_trades",
                "incremental_pool_net_pnl",
                "incremental_pool_full_sl_rate",
                "incremental_pool_timeout_rate",
            ],
            max_rows=80,
        ),
        "",
        "## Fold Detail",
        "",
        _fmt_table(
            folds_all.sort_values(["variant_name", "method", "validation_week"]) if not folds_all.empty else folds_all,
            [
                "variant_name",
                "method",
                "validation_week",
                "model_name",
                "threshold",
                "baseline_trades",
                "addon_pool_trades",
                "addon_trades",
                "addon_net_pnl",
                "combined_trades",
                "combined_net_pnl",
                "combined_full_sl_rate",
                "combined_timeout_rate",
            ],
            max_rows=140,
        ),
    ]
    paths["report"].write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps(_json_safe({"conclusion": conclusion, "outputs": {k: str(v) for k, v in paths.items()}}), indent=2))


if __name__ == "__main__":
    main()
