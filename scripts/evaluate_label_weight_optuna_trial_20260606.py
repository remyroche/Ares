#!/usr/bin/env python3
"""Evaluate one label_weight_optuna trial for the deployed long_dist strategy.

The Optuna driver writes a recipe JSON and passes it through
EPM_LABEL_WEIGHT_RECIPE. This evaluator trains only the base model for the
single deployed long_dist strategy, then writes the metrics JSON expected by
extreme_price_movements.label_weight_optuna.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

STRATEGY_ID = (
    "dist_ema20_atr_-0_92271453_loc_bb_channel_pos_48_0_60767579_"
    "leverage_build_score_0_45107844_return_autocorr_48_1_18643_"
    "rolling_range_20_-0_25967735"
)

OOF_NAME = f"oof_{STRATEGY_ID}_H5.parquet"


def _trial_number_from_recipe(path: str) -> str:
    match = re.search(r"trial_(\d+)_recipe\\.json$", str(path))
    return match.group(1) if match else "unknown"


def _base_env(run_id: str) -> dict[str, str]:
    phase = str(os.getenv("EPM_LABEL_WEIGHT_PHASE", "")).strip().lower()
    optimizing_distillation = phase == "distillation"
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": ".",
            "EPM_ARTIFACT_SOURCE_RUN_ID": "20260523_015947",
            "EPM_LGBM_NATIVE_PRESET_SOURCE_RUN_ID": "20260525_010004_nopenalty",
            "EPM_MODEL_BACKEND": "lgbm_pipeline",
            "EPM_TRAINING_NO_PENALTY": "1",
            "EPM_LGBM_USE_NATIVE_PRESET": "1",
            "EPM_LGBM_CV_SPLITS": os.getenv("EPM_LGBM_CV_SPLITS", "3"),
            "EPM_LGBM_PURGED_CV": "0",
            "EPM_LGBM_CV_MODE": os.getenv("EPM_LGBM_CV_MODE", "interleaved_spread"),
            "EPM_LGBM_RECENCY_WEIGHTING": "1",
            "EPM_LGBM_BASE_RECENCY_HALF_LIFE_DAYS": "365",
            "EPM_LGBM_META_RECENCY_HALF_LIFE_DAYS": "182.5",
            "EPM_LGBM_TRUE_SOFT_LABELS": "1",
            "EPM_LGBM_REBALANCE_EFFECTIVE_CLASSES": "1",
            "EPM_LGBM_REBALANCE_POS_MASS_MIN": "0.25",
            "EPM_LGBM_REBALANCE_POS_MASS_MAX": "0.55",
            "EPM_LGBM_REBALANCE_MAX_MULTIPLIER": "2.0",
            "EPM_TRAIN_EXTEND_TO_LATEST": "1",
            "EPM_TRAIN_RECENT_DAYS": "730",
            "EPM_BASE_STRATEGY_IDS": STRATEGY_ID,
            "EPM_META_STRATEGY_IDS": STRATEGY_ID,
            "EPM_REQUIRE_STRATEGY_ALLOWLIST": "1",
            "EPM_SKIP_MASK_STRATEGY_PARAMS": "1",
            "EPM_EXECUTION_AWARE_COST_BPS": os.getenv("EPM_EXECUTION_AWARE_COST_BPS", "68.83"),
            "EPM_LABEL_WEIGHT_USE_BEST_DEFAULT": "0",
            "EPM_LGBM_SKIP_FINAL_OOF_META_CV": "1",
            "EPM_LGBM_SKIP_REFERENCE_ARTIFACTS": "1",
            "EPM_LGBM_OPTUNA_CANDIDATE_ONLY": "1",
            "EPM_SKIP_TAIL_CONTROL_REPORTS": "1",
            "EPM_LGBM_DISABLE_SELF_DISTILLATION": "0" if optimizing_distillation else "1",
        }
    )
    env["EPM_OPTUNA_EVAL_RUN_ID"] = run_id
    return env


def _hhi(values: pd.Series) -> float:
    if values.empty:
        return 1.0
    shares = values.astype(str).value_counts(normalize=True).to_numpy(dtype=float)
    return float(np.sum(np.square(shares))) if len(shares) else 1.0


def _weighted_corr(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(weights) & (weights > 0.0)
    if int(np.sum(finite)) < 3:
        return 0.0
    x = x[finite].astype(float)
    y = y[finite].astype(float)
    weights = weights[finite].astype(float)
    weights = weights / max(float(np.sum(weights)), 1e-12)
    x_centered = x - float(np.sum(weights * x))
    y_centered = y - float(np.sum(weights * y))
    cov = float(np.sum(weights * x_centered * y_centered))
    x_var = float(np.sum(weights * x_centered * x_centered))
    y_var = float(np.sum(weights * y_centered * y_centered))
    denom = math.sqrt(max(x_var * y_var, 0.0))
    return float(cov / denom) if denom > 1e-12 else 0.0


def _weighted_spearman(score: np.ndarray, target: np.ndarray, weights: np.ndarray) -> float:
    score_rank = pd.Series(score).rank(method="average").to_numpy(dtype=float)
    target_rank = pd.Series(target).rank(method="average").to_numpy(dtype=float)
    return _weighted_corr(score_rank, target_rank, weights)


def _top_rank_weights(score_rank_pct: np.ndarray, *, top_frac: float) -> np.ndarray:
    threshold = max(0.0, min(1.0, 1.0 - float(top_frac)))
    width = max(float(top_frac), 1e-6)
    ramp = np.clip((score_rank_pct - threshold) / width, 0.0, 1.0)
    return 0.25 + 0.75 * ramp


def _topk_metrics(df: pd.DataFrame, *, start: pd.Timestamp | None, cost_bps: float, k: int) -> dict[str, float]:
    d = df if start is None else df[df["timestamp"] >= start]
    d = d[np.isfinite(d["oof_prob"]) & np.isfinite(d["y_ret"])]
    if d.empty:
        return {
            "n": 0.0,
            "net_hit": 0.0,
            "mean_net_bps": 0.0,
            "median_net_bps": 0.0,
            "stop_hit_rate": 1.0,
            "avg_stop_loss_bps": 0.0,
            "unique_symbols": 0.0,
            "symbol_hhi": 1.0,
            "unique_weeks": 0.0,
            "week_hhi": 1.0,
        }
    top = (
        d.sort_values(["timestamp", "oof_prob"], ascending=[True, False])
        .groupby("timestamp", sort=False)
        .head(k)
    )
    net_bps = top["y_ret"].to_numpy(dtype=float) * 10_000.0 - cost_bps
    mae_bps = top.get("mae_ret", pd.Series(np.zeros(len(top)), index=top.index)).to_numpy(dtype=float) * 10_000.0
    stop = mae_bps > 100.0
    stop_losses = np.maximum(-net_bps[stop], 0.0)
    weeks = pd.to_datetime(top["timestamp"], utc=True).dt.tz_convert(None).dt.to_period("W").astype(str)
    return {
        "n": float(len(top)),
        "net_hit": float(np.mean(net_bps > 0.0)),
        "mean_net_bps": float(np.nanmean(net_bps)),
        "median_net_bps": float(np.nanmedian(net_bps)),
        "stop_hit_rate": float(np.mean(stop)),
        "avg_stop_loss_bps": float(np.nanmean(stop_losses)) if len(stop_losses) else 0.0,
        "unique_symbols": float(top["symbol"].nunique()) if "symbol" in top else 0.0,
        "symbol_hhi": _hhi(top["symbol"]) if "symbol" in top else 1.0,
        "unique_weeks": float(weeks.nunique()),
        "week_hhi": _hhi(weeks),
    }


def _ranking_surface_metrics(df: pd.DataFrame, *, start: pd.Timestamp | None, cost_bps: float) -> dict[str, float]:
    d = df if start is None else df[df["timestamp"] >= start]
    d = d[np.isfinite(d["oof_prob"]) & np.isfinite(d["y_ret"])]
    if d.empty:
        return {
            "prediction_score_std": 0.0,
            "prediction_score_iqr": 0.0,
            "score_gap_top10_to_30_40": 0.0,
            "economic_rank_ic": 0.0,
            "economic_weighted_ic": 0.0,
            "economic_weighted_ic_full": 0.0,
            "economic_weighted_ic_top30": 0.0,
            "economic_weighted_ic_top20": 0.0,
            "economic_weighted_ic_top10": 0.0,
            "economic_rank_monotonicity": 0.5,
            "economic_bucket_spread_bps": 0.0,
        }
    score = d["oof_prob"].to_numpy(dtype=float)
    net_bps = d["y_ret"].to_numpy(dtype=float) * 10_000.0 - cost_bps
    finite = np.isfinite(score) & np.isfinite(net_bps)
    score = score[finite]
    net_bps = net_bps[finite]
    if len(score) < 10:
        return {
            "prediction_score_std": float(np.nanstd(score)) if len(score) else 0.0,
            "prediction_score_iqr": 0.0,
            "score_gap_top10_to_30_40": 0.0,
            "economic_rank_ic": 0.0,
            "economic_weighted_ic": 0.0,
            "economic_weighted_ic_full": 0.0,
            "economic_weighted_ic_top30": 0.0,
            "economic_weighted_ic_top20": 0.0,
            "economic_weighted_ic_top10": 0.0,
            "economic_rank_monotonicity": 0.5,
            "economic_bucket_spread_bps": 0.0,
        }
    ranks = pd.Series(score).rank(pct=True, method="average").to_numpy(dtype=float)
    top10 = score[ranks >= 0.90]
    mid30_40 = score[(ranks >= 0.60) & (ranks < 0.70)]
    score_gap = (
        float(np.nanmean(top10) - np.nanmean(mid30_40))
        if len(top10) and len(mid30_40)
        else 0.0
    )
    rank_ic = pd.Series(score).corr(pd.Series(net_bps), method="spearman")
    full_weights = 0.50 + 0.50 * ranks
    weighted_ic_full = _weighted_spearman(score, net_bps, full_weights)
    weighted_ic_top30 = _weighted_spearman(score, net_bps, _top_rank_weights(ranks, top_frac=0.30))
    weighted_ic_top20 = _weighted_spearman(score, net_bps, _top_rank_weights(ranks, top_frac=0.20))
    weighted_ic_top10 = _weighted_spearman(score, net_bps, _top_rank_weights(ranks, top_frac=0.10))
    weighted_ic = (
        0.25 * weighted_ic_full
        + 0.20 * weighted_ic_top30
        + 0.15 * weighted_ic_top20
        + 0.10 * weighted_ic_top10
    ) / 0.70
    try:
        bucket = pd.qcut(score, q=min(10, len(np.unique(score))), labels=False, duplicates="drop")
    except ValueError:
        bucket = None
    if bucket is None or pd.isna(bucket).all():
        monotonicity = 0.5
        bucket_spread = 0.0
    else:
        by_bucket = pd.DataFrame({"bucket": bucket, "net_bps": net_bps}).dropna()
        means = by_bucket.groupby("bucket", sort=True)["net_bps"].mean().to_numpy(dtype=float)
        if len(means) < 2:
            monotonicity = 0.5
            bucket_spread = 0.0
        else:
            diffs = np.diff(means)
            monotonicity = float(np.mean(diffs >= 0.0))
            bucket_spread = float(means[-1] - means[0])
    return {
        "prediction_score_std": float(np.nanstd(score)),
        "prediction_score_iqr": float(np.nanpercentile(score, 75) - np.nanpercentile(score, 25)),
        "score_gap_top10_to_30_40": score_gap,
        "economic_rank_ic": float(rank_ic) if rank_ic is not None and math.isfinite(float(rank_ic)) else 0.0,
        "economic_weighted_ic": float(weighted_ic),
        "economic_weighted_ic_full": float(weighted_ic_full),
        "economic_weighted_ic_top30": float(weighted_ic_top30),
        "economic_weighted_ic_top20": float(weighted_ic_top20),
        "economic_weighted_ic_top10": float(weighted_ic_top10),
        "economic_rank_monotonicity": monotonicity,
        "economic_bucket_spread_bps": bucket_spread,
    }


def _unit_interval(value: float, *, floor: float, good: float) -> float:
    if not math.isfinite(float(value)) or abs(good - floor) <= 1e-12:
        return 0.0
    return float(np.clip((float(value) - floor) / (good - floor), 0.0, 1.0))


def _lgbm_style_j_proxy(metrics: dict[str, float]) -> float:
    ic_component = _unit_interval(float(metrics.get("economic_weighted_ic", 0.0)), floor=-0.02, good=0.12)
    mono_component = float(np.clip(float(metrics.get("economic_rank_monotonicity", 0.5)), 0.0, 1.0))
    spread_component = _unit_interval(float(metrics.get("economic_bucket_spread_bps", 0.0)), floor=-25.0, good=100.0)
    std_component = _unit_interval(float(metrics.get("prediction_score_std", 0.0)), floor=0.01, good=0.10)
    return float(0.25 + 0.30 * ic_component + 0.25 * mono_component + 0.12 * spread_component + 0.08 * std_component)


def _score_oof(run_id: str) -> dict[str, float]:
    oof_path = ROOT / "data_perp" / "artifacts" / run_id / "oof" / OOF_NAME
    if not oof_path.exists():
        raise FileNotFoundError(f"OOF parquet missing after trial training: {oof_path}")
    df = pd.read_parquet(
        oof_path,
        columns=["timestamp", "symbol", "oof_prob", "y_ret", "mfe_ret", "mae_ret"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    max_ts = df["timestamp"].max()
    cost_bps = float(os.getenv("EPM_EXECUTION_AWARE_COST_BPS", "68.83"))
    windows = {
        "full": None,
        "26w": max_ts - pd.Timedelta(weeks=26),
        "13w": max_ts - pd.Timedelta(weeks=13),
        "8w": max_ts - pd.Timedelta(weeks=8),
    }
    per_window: dict[str, dict[str, dict[str, float]]] = {}
    per_window_ranking: dict[str, dict[str, float]] = {}
    for win, start in windows.items():
        per_window[win] = {}
        for k in (10, 20, 30, 50):
            per_window[win][str(k)] = _topk_metrics(df, start=start, cost_bps=cost_bps, k=k)
        per_window_ranking[win] = _ranking_surface_metrics(df, start=start, cost_bps=cost_bps)

    # Optimise for execution quality that survives broader temporal slices.
    weights = {"full": 0.20, "26w": 0.30, "13w": 0.30, "8w": 0.20}
    metrics: dict[str, float] = {"model_stage": "base", "execution_cost_bps": cost_bps}
    for k in (10, 20, 30, 50):
        metrics[f"net_hit_at_{k}"] = float(
            sum(weights[w] * per_window[w][str(k)]["net_hit"] for w in weights)
        )
        metrics[f"mean_net_bps_at_{k}"] = float(
            sum(weights[w] * per_window[w][str(k)]["mean_net_bps"] for w in weights)
        )
        metrics[f"median_net_bps_at_{k}"] = float(
            sum(weights[w] * per_window[w][str(k)]["median_net_bps"] for w in weights)
        )
        metrics[f"unique_symbols_at_{k}"] = float(
            sum(weights[w] * per_window[w][str(k)]["unique_symbols"] for w in weights)
        )
        metrics[f"symbol_concentration_hhi_at_{k}"] = float(
            sum(weights[w] * per_window[w][str(k)]["symbol_hhi"] for w in weights)
        )
        metrics[f"unique_weeks_at_{k}"] = float(
            sum(weights[w] * per_window[w][str(k)]["unique_weeks"] for w in weights)
        )
        metrics[f"week_concentration_hhi_at_{k}"] = float(
            sum(weights[w] * per_window[w][str(k)]["week_hhi"] for w in weights)
        )
    metrics["stop_hit_rate_at_20"] = float(
        sum(weights[w] * per_window[w]["20"]["stop_hit_rate"] for w in weights)
    )
    metrics["avg_stop_loss_bps_at_20"] = float(
        sum(weights[w] * per_window[w]["20"]["avg_stop_loss_bps"] for w in weights)
    )
    hit20_values = [per_window[w]["20"]["net_hit"] for w in weights]
    metrics["prediction_instability"] = float(np.nanstd(hit20_values))
    for name in (
        "prediction_score_std",
        "prediction_score_iqr",
        "score_gap_top10_to_30_40",
        "economic_rank_ic",
        "economic_weighted_ic",
        "economic_weighted_ic_full",
        "economic_weighted_ic_top30",
        "economic_weighted_ic_top20",
        "economic_weighted_ic_top10",
        "economic_rank_monotonicity",
        "economic_bucket_spread_bps",
    ):
        metrics[name] = float(sum(weights[w] * per_window_ranking[w][name] for w in weights))
    j_proxy = _lgbm_style_j_proxy(metrics)
    metrics["J_base"] = j_proxy
    metrics["J_final"] = j_proxy
    metrics["J_proxy"] = j_proxy
    metrics["J_source"] = "label_weight_eval_proxy"  # type: ignore[assignment]
    metrics["per_window"] = per_window  # type: ignore[assignment]
    metrics["per_window_ranking"] = per_window_ranking  # type: ignore[assignment]
    metrics["run_id"] = run_id  # type: ignore[assignment]
    metrics["oof_path"] = str(oof_path)  # type: ignore[assignment]
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-json", default=os.getenv("EPM_LABEL_WEIGHT_METRICS_JSON", ""))
    parser.add_argument("--run-id-prefix", default="20260606_label_weight_optuna_long_dist")
    args = parser.parse_args()
    if not args.metrics_json:
        raise SystemExit("--metrics-json is required unless EPM_LABEL_WEIGHT_METRICS_JSON is set")

    recipe_path = os.getenv("EPM_LABEL_WEIGHT_RECIPE", "")
    trial = os.getenv("EPM_LABEL_WEIGHT_TRIAL_NUMBER") or _trial_number_from_recipe(recipe_path)
    run_id = f"{args.run_id_prefix}_trial_{trial}"
    log_path = LOG_DIR / f"train_{run_id}.log"
    cmd = [
        sys.executable,
        "-u",
        "extreme_price_movements/run_pipeline.py",
        "train",
        "--base-only",
        "--market-mode",
        "perps",
        "--exchange",
        "kraken",
        "--ts",
        "20260523_015947",
        "--run-id",
        run_id,
    ]
    env = _base_env(run_id)
    with log_path.open("ab", buffering=0) as log_fp:
        log_fp.write(f"\n=== START label_weight_optuna trial={trial} run_id={run_id} ===\n".encode())
        log_fp.write(f"EPM_LABEL_WEIGHT_RECIPE={recipe_path}\n".encode())
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=log_fp, stderr=subprocess.STDOUT)
        log_fp.write(f"\n=== END label_weight_optuna trial={trial} ret={proc.returncode} ===\n".encode())
    if proc.returncode != 0:
        return int(proc.returncode)
    metrics = _score_oof(run_id)
    out = Path(args.metrics_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
