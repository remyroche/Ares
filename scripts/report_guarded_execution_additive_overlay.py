#!/usr/bin/env python3
"""Baseline-preserving additive overlay diagnostic for guarded execution.

The current anchored adaptive policy is profitable but narrow. Prior breadth
experiments widened the source and often displaced good current-anchor trades.
This diagnostic keeps current-anchor accepted trades fixed, then tests whether
incremental accepted candidates from broader variants can be added by a simple
prior-fold threshold gate without violating basic portfolio capacity.

This is not a promotion artifact: it is a diagnostic for whether incremental
breadth is learnably useful once the current champion is preserved.
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


REPORT_ROOT = Path("data_perp/reports/gmm_cluster_policy_smoke_20260702_wide_sidebalanced")
BASELINE_DIR = REPORT_ROOT / "meta_handoff_adaptive_scenario_guard_anchoradmit_h9_buffer000_v3"
DEFAULT_OUT_DIR = REPORT_ROOT / "guarded_execution_additive_overlay_20260703_v1"
DEFAULT_VARIANTS = [
    ("rank75_bad55_to12h", REPORT_ROOT / "meta_handoff_adaptive_scenario_guard_anchoradmit_h9_buffer000_rank75bad55_to12h_v1"),
    ("rank85_bad45_to12h", REPORT_ROOT / "meta_handoff_adaptive_scenario_guard_anchoradmit_h9_buffer000_rank85bad45_to12h_v1"),
    ("rank90_bad45_to12h", REPORT_ROOT / "meta_handoff_adaptive_scenario_guard_anchoradmit_h9_buffer000_rank90bad45_to12h_v1"),
    ("rank80_bad65_to12h", REPORT_ROOT / "meta_handoff_adaptive_scenario_guard_anchoradmit_h9_buffer000_rank80bad65_to12h_v1"),
    ("min_train_trades_26", REPORT_ROOT / "meta_handoff_adaptive_scenario_guard_anchoradmit_h9_buffer000_mintrades26_v1"),
    ("min_train_trades_30", REPORT_ROOT / "meta_handoff_adaptive_scenario_guard_anchoradmit_h9_buffer000_mintrades30_v1"),
    ("wide_grid_min_train_trades_30", REPORT_ROOT / "meta_handoff_adaptive_scenario_guard_anchoradmit_h9_buffer000_widegrid_mintrades30_v1"),
]
SCORE_COLUMNS = (
    "adaptive_guard_margin",
    "adaptive_guard_score_oof",
    "effective_rank_score",
    "normalized_rank_score",
    "rank_minus_joint_bad",
    "rank_minus_joint_timeout",
    "oof_regime_centroid_similarity_train",
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
)
CONTEXT_COLUMNS = (
    "base_opportunity_key",
    "scenario",
    "rank_minus_joint_bad",
    "rank_minus_joint_timeout",
    "oof_regime_centroid_similarity_train",
    "archetype_meta_bad_risk",
    "archetype_meta_timeout_risk",
    "archetype_joint_bad_risk",
    "archetype_joint_timeout_risk",
    "horizon_hours",
    "barrier_multiplier",
    "policy_effective_barrier_pct",
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _load_accepted(path: Path) -> pd.DataFrame:
    decisions = pd.read_parquet(path)
    if decisions.empty:
        return pd.DataFrame()
    mask = (
        decisions["accepted"].astype(bool)
        if "accepted" in decisions.columns
        else pd.Series(True, index=decisions.index)
    )
    out = decisions.loc[mask].copy()
    if out.empty:
        return pd.DataFrame()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["position_exit_timestamp"] = pd.to_datetime(
        out.get("position_exit_timestamp"),
        utc=True,
        errors="coerce",
    )
    out["validation_week"] = out.get("validation_week", "").astype(str)
    out["symbol"] = out.get("symbol", "").astype(str)
    out["side"] = out.get("side", "").astype(str)
    out["strategy_id"] = out.get("strategy_id", "").astype(str)
    if "base_opportunity_key" not in out.columns:
        out["base_opportunity_key"] = (
            out["timestamp"].astype(str) + "|" + out["symbol"] + "|" + out["side"] + "|" + out["strategy_id"]
        )
    out["base_opportunity_key"] = out["base_opportunity_key"].astype(str)
    out["accepted_key"] = (
        out["timestamp"].astype(str) + "|" + out["symbol"] + "|" + out["side"] + "|" + out["strategy_id"]
    )
    out["accepted_net_return"] = pd.to_numeric(out.get("position_net_return", 0.0), errors="coerce").fillna(0.0)
    out["accepted_gross_return"] = pd.to_numeric(out.get("position_gross_return", 0.0), errors="coerce").fillna(0.0)
    out["position_size"] = pd.to_numeric(out.get("position_size", 0.0), errors="coerce").fillna(0.0)
    out["accepted_net_pnl"] = out["position_size"] * out["accepted_net_return"]
    reason = out.get("position_exit_reason", out.get("simple_policy_exit_reason", pd.Series("", index=out.index))).astype(str)
    out["accepted_exit_reason"] = reason
    out["is_full_sl"] = reason.eq("full_sl")
    out["is_timeout"] = reason.eq("timeout")
    out["side_name"] = np.where(out["side"].astype(str).str.lower().str.contains("short"), "short", "long")
    for col in SCORE_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.sort_values(["timestamp", "symbol", "strategy_id"]).reset_index(drop=True)


def _load_context(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    if frame.empty or "base_opportunity_key" not in frame.columns:
        return pd.DataFrame()
    keep = [col for col in CONTEXT_COLUMNS if col in frame.columns]
    context = frame[keep].copy()
    for col in keep:
        if col not in {"base_opportunity_key", "scenario"}:
            context[col] = pd.to_numeric(context[col], errors="coerce")
    for risk_col in (
        "archetype_meta_bad_risk",
        "archetype_meta_timeout_risk",
        "archetype_joint_bad_risk",
        "archetype_joint_timeout_risk",
    ):
        if risk_col in context.columns:
            context[f"neg_{risk_col}"] = -pd.to_numeric(context[risk_col], errors="coerce")
    keys = ["base_opportunity_key"]
    if "scenario" in context.columns:
        keys.append("scenario")
    return context.drop_duplicates(keys).reset_index(drop=True)


def _load_variant_accepted(variant_dir: Path) -> pd.DataFrame:
    accepted = _load_accepted(variant_dir / "adaptive_scenario_guard_decisions.parquet")
    if accepted.empty:
        return accepted
    context = _load_context(variant_dir / "adaptive_scenario_guard_selected_candidates.parquet")
    if context.empty:
        return accepted
    keys = ["base_opportunity_key"]
    if "scenario" in accepted.columns and "scenario" in context.columns:
        keys.append("scenario")
    merged = accepted.merge(context, on=keys, how="left", suffixes=("", "_context"))
    return _add_composite_scores(merged)


def _add_composite_scores(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()

    def numeric(col: str) -> pd.Series:
        return pd.to_numeric(out[col], errors="coerce") if col in out.columns else pd.Series(np.nan, index=out.index)

    joint_bad = numeric("archetype_joint_bad_risk")
    meta_bad = numeric("archetype_meta_bad_risk")
    joint_timeout = numeric("archetype_joint_timeout_risk")
    out["effective_rank_minus_joint_bad"] = numeric("effective_rank_score") - joint_bad
    out["effective_rank_minus_meta_bad"] = numeric("effective_rank_score") - meta_bad
    out["effective_rank_minus_joint_bad_timeout"] = numeric("effective_rank_score") - joint_bad - 0.5 * joint_timeout
    out["guard_score_minus_joint_bad"] = numeric("adaptive_guard_score_oof") - joint_bad
    out["guard_margin_minus_joint_bad"] = numeric("adaptive_guard_margin") - joint_bad
    out["calibrated_minus_joint_bad"] = numeric("calibrated_score") - joint_bad
    out["neg_joint_bad_timeout_sum"] = -(joint_bad + joint_timeout)
    return out


def _candidate_incremental(variant: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    baseline_keys = set(baseline["base_opportunity_key"].astype(str))
    baseline_accepted_keys = set(baseline["accepted_key"].astype(str))
    out = variant.loc[
        ~variant["base_opportunity_key"].astype(str).isin(baseline_keys)
        & ~variant["accepted_key"].astype(str).isin(baseline_accepted_keys)
    ].copy()
    return out.reset_index(drop=True)


def _portfolio_capacity_filter(
    *,
    baseline: pd.DataFrame,
    add_ons: pd.DataFrame,
    max_open: int,
    max_open_per_side: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Greedily add candidates without overlapping baseline/add-on capacity."""
    if add_ons.empty:
        return add_ons.copy(), pd.DataFrame()
    fixed_positions = baseline[
        [
            "timestamp",
            "position_exit_timestamp",
            "symbol",
            "side_name",
            "base_opportunity_key",
        ]
    ].copy()
    open_positions: list[dict[str, Any]] = fixed_positions.to_dict(orient="records")
    accepted_rows: list[pd.Series] = []
    rejected_rows: list[dict[str, Any]] = []
    for _, row in add_ons.sort_values(["timestamp", "effective_rank_score", "accepted_net_return"], ascending=[True, False, False]).iterrows():
        ts = row["timestamp"]
        if pd.isna(ts):
            continue
        active = [
            pos
            for pos in open_positions
            if pd.notna(pos.get("timestamp"))
            and pd.notna(pos.get("position_exit_timestamp"))
            and pos["timestamp"] <= ts < pos["position_exit_timestamp"]
        ]
        symbol_open = any(str(pos.get("symbol")) == str(row["symbol"]) for pos in active)
        side_open = sum(1 for pos in active if str(pos.get("side_name")) == str(row["side_name"]))
        reason = ""
        if symbol_open:
            reason = "symbol_overlap"
        elif len(active) >= int(max_open):
            reason = "max_open"
        elif side_open >= int(max_open_per_side):
            reason = "max_side"
        if reason:
            rejected_rows.append(
                {
                    "timestamp": ts,
                    "symbol": row["symbol"],
                    "side": row["side"],
                    "base_opportunity_key": row["base_opportunity_key"],
                    "rejection_reason": reason,
                }
            )
            continue
        accepted_rows.append(row)
        open_positions.append(
            {
                "timestamp": row["timestamp"],
                "position_exit_timestamp": row["position_exit_timestamp"],
                "symbol": row["symbol"],
                "side_name": row["side_name"],
                "base_opportunity_key": row["base_opportunity_key"],
            }
        )
    accepted = pd.DataFrame(accepted_rows).reset_index(drop=True) if accepted_rows else pd.DataFrame(columns=add_ons.columns)
    rejected = pd.DataFrame(rejected_rows)
    return accepted, rejected


def _threshold_grid(values: pd.Series) -> list[float]:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if numeric.empty:
        return []
    qs = [0.0, 0.10, 0.25, 0.35, 0.50, 0.65, 0.75, 0.90]
    return sorted({float(numeric.quantile(q)) for q in qs})


def _score_rows(rows: pd.DataFrame) -> dict[str, float]:
    if rows.empty:
        return {
            "trades": 0,
            "net_pnl": 0.0,
            "mean_return": np.nan,
            "hit_rate": np.nan,
            "full_sl_rate": np.nan,
            "timeout_rate": np.nan,
        }
    return {
        "trades": int(len(rows)),
        "net_pnl": float(rows["accepted_net_pnl"].sum()),
        "mean_return": float(rows["accepted_net_return"].mean()),
        "hit_rate": float(rows["accepted_net_return"].gt(0.0).mean()),
        "full_sl_rate": float(rows["is_full_sl"].mean()),
        "timeout_rate": float(rows["is_timeout"].mean()),
    }


def _objective(stats: dict[str, float], *, min_trades: int) -> float:
    if int(stats["trades"]) < int(min_trades):
        return -1.0e9 + float(stats["trades"])
    return float(
        0.00002 * stats["net_pnl"]
        + (0.0 if np.isnan(stats["mean_return"]) else stats["mean_return"])
        - 0.08 * (0.0 if np.isnan(stats["full_sl_rate"]) else stats["full_sl_rate"])
        - 0.03 * (0.0 if np.isnan(stats["timeout_rate"]) else stats["timeout_rate"])
    )


def _select_threshold(
    train: pd.DataFrame,
    *,
    score_col: str,
    min_train_addons: int,
    require_positive_train_score: bool,
) -> tuple[float, float, dict[str, float]]:
    if train.empty or score_col not in train.columns:
        return float("inf"), 0.0, _score_rows(pd.DataFrame())
    best: tuple[float, float, dict[str, float]] | None = (
        float("inf"),
        0.0,
        _score_rows(pd.DataFrame()),
    )
    for threshold in _threshold_grid(train[score_col]):
        kept = train.loc[pd.to_numeric(train[score_col], errors="coerce").ge(float(threshold))]
        stats = _score_rows(kept)
        score = _objective(stats, min_trades=int(min_train_addons))
        current = (float(threshold), float(score), stats)
        if best is None or current[1] > best[1]:
            best = current
    if require_positive_train_score and best[1] <= 0.0:
        return float("inf"), float(best[1]), _score_rows(pd.DataFrame())
    return best if best is not None else (float("inf"), -1.0e9, _score_rows(pd.DataFrame()))


def _fold_overlay(
    baseline: pd.DataFrame,
    incremental: pd.DataFrame,
    *,
    score_col: str,
    min_train_addons: int,
    max_open: int,
    max_open_per_side: int,
    require_positive_train_score: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    kept_frames: list[pd.DataFrame] = []
    weeks = sorted(baseline["validation_week"].dropna().astype(str).unique())
    for fold_id, week in enumerate(weeks):
        base_eval = baseline.loc[baseline["validation_week"].astype(str).eq(week)].copy()
        train = incremental.loc[incremental["validation_week"].astype(str) < str(week)].copy()
        eval_pool = incremental.loc[incremental["validation_week"].astype(str).eq(week)].copy()
        threshold, train_score, train_stats = _select_threshold(
            train,
            score_col=score_col,
            min_train_addons=int(min_train_addons),
            require_positive_train_score=bool(require_positive_train_score),
        )
        if not np.isfinite(float(threshold)):
            pre_capacity = eval_pool.iloc[0:0].copy()
        else:
            eval_scores = pd.to_numeric(eval_pool.get(score_col), errors="coerce").replace(
                [np.inf, -np.inf],
                np.nan,
            )
            pre_capacity = eval_pool.loc[eval_scores.ge(float(threshold)).fillna(False)].copy()
        add_eval, rejected_capacity = _portfolio_capacity_filter(
            baseline=baseline.loc[baseline["validation_week"].astype(str).le(str(week))].copy(),
            add_ons=pre_capacity,
            max_open=int(max_open),
            max_open_per_side=int(max_open_per_side),
        )
        if not add_eval.empty:
            add_eval["overlay_score_col"] = score_col
            add_eval["overlay_threshold"] = float(threshold)
            add_eval["overlay_train_score"] = float(train_score)
            kept_frames.append(add_eval)
        combined = pd.concat([base_eval, add_eval], ignore_index=True) if not add_eval.empty else base_eval
        base_stats = _score_rows(base_eval)
        add_stats = _score_rows(add_eval)
        combined_stats = _score_rows(combined)
        rows.append(
            {
                "fold_id": int(fold_id),
                "validation_week": week,
                "score_col": score_col,
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


def _summarise_folds(folds: pd.DataFrame, *, variant: str, score_col: str) -> dict[str, Any]:
    if folds.empty:
        return {"variant_name": variant, "score_col": score_col, "status": "fail"}
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
        "score_col": score_col,
        "status": "pass" if pass_gate else "fail",
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


def _fmt_table(frame: pd.DataFrame, cols: list[str], max_rows: int = 40) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame[[col for col in cols if col in frame.columns]].head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:,.6f}")
    return view.to_markdown(index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=BASELINE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-train-addons", type=int, default=3)
    parser.add_argument("--max-open", type=int, default=8)
    parser.add_argument("--max-open-per-side", type=int, default=4)
    parser.add_argument(
        "--allow-negative-train-score",
        action="store_true",
        help="Permit add-ons when every prior-fold threshold has non-positive selector objective.",
    )
    args = parser.parse_args()

    baseline = _load_variant_accepted(args.baseline_dir)
    if baseline.empty:
        raise ValueError(f"Missing baseline accepted decisions under {args.baseline_dir}")

    summary_rows: list[dict[str, Any]] = []
    fold_frames: list[pd.DataFrame] = []
    kept_frames: list[pd.DataFrame] = []
    incremental_rows: list[dict[str, Any]] = []
    for variant_name, variant_dir in DEFAULT_VARIANTS:
        variant = _load_variant_accepted(variant_dir)
        if variant.empty:
            continue
        incremental = _candidate_incremental(variant, baseline)
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
        for score_col in SCORE_COLUMNS:
            if score_col not in incremental.columns:
                continue
            folds, kept = _fold_overlay(
                baseline,
                incremental,
                score_col=score_col,
                min_train_addons=int(args.min_train_addons),
                max_open=int(args.max_open),
                max_open_per_side=int(args.max_open_per_side),
                require_positive_train_score=not bool(args.allow_negative_train_score),
            )
            if not folds.empty:
                folds["variant_name"] = variant_name
                fold_frames.append(folds)
            if not kept.empty:
                kept["variant_name"] = variant_name
                kept_frames.append(kept)
            summary_rows.append(_summarise_folds(folds, variant=variant_name, score_col=score_col))

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
        "status": "overlay_candidate_found" if pass_count else "no_overlay_candidate",
        "overlay_gate_pass_count": pass_count,
        "best_variant": best.get("variant_name", ""),
        "best_score_col": best.get("score_col", ""),
        "best_combined_trades": int(best.get("combined_trades", 0) or 0),
        "best_addon_trades": int(best.get("addon_trades", 0) or 0),
        "best_combined_net_pnl": float(best.get("combined_net_pnl", np.nan)) if best else np.nan,
        "interpretation": (
            "Baseline-preserving add-ons did not beat the current anchor under prior-fold score thresholds."
            if not pass_count
            else "A baseline-preserving add-on candidate cleared the diagnostic gate; it still requires materialized frozen replay."
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": args.out_dir / "guarded_execution_additive_overlay_summary.csv",
        "folds": args.out_dir / "guarded_execution_additive_overlay_folds.csv",
        "incremental_pool": args.out_dir / "guarded_execution_additive_overlay_incremental_pool.csv",
        "kept_addons": args.out_dir / "guarded_execution_additive_overlay_kept_addons.parquet",
        "manifest": args.out_dir / "guarded_execution_additive_overlay_manifest.json",
        "report": args.out_dir / "guarded_execution_additive_overlay_report.md",
    }
    summary.to_csv(paths["summary"], index=False)
    folds_all.to_csv(paths["folds"], index=False)
    incremental.to_csv(paths["incremental_pool"], index=False)
    kept_all.to_parquet(paths["kept_addons"], index=False)
    manifest = {
        "generated_by": "report_guarded_execution_additive_overlay",
        "out_dir": str(args.out_dir),
        "baseline_dir": str(args.baseline_dir),
        "variants": [{"name": name, "directory": str(path)} for name, path in DEFAULT_VARIANTS],
        "score_columns": list(SCORE_COLUMNS),
        "min_train_addons": int(args.min_train_addons),
        "max_open": int(args.max_open),
        "max_open_per_side": int(args.max_open_per_side),
        "require_positive_train_score": not bool(args.allow_negative_train_score),
        "diagnostic_only": True,
        "diagnostic_caveat": "Baseline trades are preserved and add-ons are capacity-filtered, but this is not a full frozen portfolio replay.",
        "conclusion": conclusion,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    report_lines = [
        "# Guarded Execution Additive Overlay Diagnostic",
        "",
        f"Status: `{conclusion['status']}`",
        "",
        "## Conclusion",
        "",
        str(conclusion["interpretation"]),
        "",
        "## Summary",
        "",
        _fmt_table(
            summary,
            [
                "variant_name",
                "score_col",
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
            folds_all.sort_values(["variant_name", "score_col", "validation_week"]) if not folds_all.empty else folds_all,
            [
                "variant_name",
                "score_col",
                "validation_week",
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
            max_rows=120,
        ),
    ]
    paths["report"].write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps(_json_safe({"conclusion": conclusion, "outputs": {k: str(v) for k, v in paths.items()}}), indent=2))


if __name__ == "__main__":
    main()
