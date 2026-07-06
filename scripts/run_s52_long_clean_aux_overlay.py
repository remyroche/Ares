#!/usr/bin/env python3
"""Blend a long clean-path auxiliary head into the S52 clean-gross ranker."""

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

try:
    from lightgbm import LGBMClassifier, LGBMRegressor

    _LIGHTGBM_AVAILABLE = True
except Exception:  # pragma: no cover
    LGBMClassifier = None
    LGBMRegressor = None
    _LIGHTGBM_AVAILABLE = False

from scripts.diagnose_s52_long_clean_dirty_separability import _candidate_labels  # noqa: E402
from scripts.run_gate3_side_soft_label_hpo import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    DEFAULT_FEATURE_LIST_CSV,
    DEFAULT_ROUND_TRIP_COST,
    _json_safe,
    _prepare_folds,
    _score_fold,
    _summarize_trial,
)
from scripts.run_s52_ranker_smoke import (  # noqa: E402
    DEFAULT_BEST_CONFIG,
    DEFAULT_LABELS_PATH,
    DEFAULT_MONTHS,
    _load_config,
    _materialized_soft_label,
    _parse_csv,
    _ranker_sample_weight,
    _fit_ranker,
    _scored_ledger,
)


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/s52_long_clean_aux_overlay_20260705_v1")


def _parse_float_csv(value: str | None, default: tuple[float, ...]) -> list[float]:
    text = str(value or "").strip()
    if not text:
        return list(default)
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def _cap_indices(n_rows: int, max_rows: int, seed: int) -> np.ndarray:
    if int(max_rows) <= 0 or n_rows <= int(max_rows):
        return np.arange(n_rows, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(n_rows, size=int(max_rows), replace=False)
    return np.sort(idx.astype(np.int64))


def _rank_pct(values: np.ndarray) -> np.ndarray:
    s = pd.Series(values).replace([np.inf, -np.inf], np.nan)
    return s.rank(method="average", pct=True).fillna(0.0).to_numpy(dtype=np.float32)


def _long_clean_gross_target(
    labels: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    round_trip_cost: float,
    mode: str,
) -> pd.Series:
    clean = labels["clean"].astype(float).reset_index(drop=True)
    gross = pd.to_numeric(labels["gross"], errors="coerce").fillna(0.0).reset_index(drop=True)
    ts = pd.to_datetime(frame["__ts__"], errors="coerce").reset_index(drop=True)
    gross_positive = gross.clip(lower=0.0)
    gross_rank = gross_positive.groupby(ts, dropna=False).rank(method="average", pct=True).fillna(0.0)
    mode_norm = str(mode or "clean_gross_rank").strip().lower()
    if mode_norm == "clean_gross_rank":
        target = clean * gross.gt(0.0).astype(float) * (0.35 + 0.65 * gross_rank)
    elif mode_norm == "clean_gross_margin":
        cost_scale = max(float(round_trip_cost), 1e-6)
        gross_strength = (gross / cost_scale).clip(lower=0.0, upper=2.0) / 2.0
        target = clean * gross.gt(0.0).astype(float) * (0.50 * gross_rank + 0.50 * gross_strength)
    else:
        raise ValueError(f"unknown EV-aware aux target mode: {mode}")
    return target.clip(0.0, 1.0).astype(np.float32)


def _fit_long_aux_score(
    *,
    fold: dict[str, Any],
    round_trip_cost: float,
    max_aux_train_rows: int,
    seed: int,
    aux_mode: str,
) -> np.ndarray:
    if not _LIGHTGBM_AVAILABLE or LGBMClassifier is None or LGBMRegressor is None:
        raise RuntimeError("lightgbm is required")
    train_labels = _candidate_labels(
        fold["train_frame"],
        fold["train_metrics"],
        round_trip_cost=round_trip_cost,
        side="long",
    )
    train_idx_all = np.flatnonzero(train_labels["candidate"].to_numpy(dtype=bool))
    train_idx = train_idx_all[_cap_indices(len(train_idx_all), int(max_aux_train_rows), seed=int(seed))]
    long_valid = (
        pd.to_numeric(fold["valid_metrics"].get("side", pd.Series(1.0, index=fold["valid_metrics"].index)), errors="coerce")
        .fillna(1.0)
        .ge(0.0)
        .reset_index(drop=True)
    )
    score = np.full(int(fold["valid_rows"]), 0.5, dtype=np.float32)
    if len(train_idx) < 500 or train_labels.iloc[train_idx]["clean"].nunique() < 2 or int(long_valid.sum()) == 0:
        return score
    x_train = fold["x_train"].iloc[train_idx].reset_index(drop=True)
    valid_idx = np.flatnonzero(long_valid.to_numpy(dtype=bool))
    aux_mode_norm = str(aux_mode or "clean_prob").strip().lower()
    if aux_mode_norm == "clean_prob":
        y_train = train_labels.iloc[train_idx]["clean"].astype(int).reset_index(drop=True)
        model = LGBMClassifier(
            objective="binary",
            n_estimators=180,
            learning_rate=0.035,
            num_leaves=31,
            min_child_samples=70,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=5.0,
            random_state=int(seed),
            n_jobs=2,
            verbosity=-1,
        )
        weights = np.where(y_train.to_numpy(dtype=bool), 1.75, 1.0).astype(np.float32)
        model.fit(x_train, y_train, sample_weight=weights)
        score[valid_idx] = model.predict_proba(fold["x_valid"].iloc[valid_idx].reset_index(drop=True))[:, 1].astype(
            np.float32
        )
        return score

    target_all = _long_clean_gross_target(
        train_labels,
        fold["train_frame"].reset_index(drop=True),
        round_trip_cost=float(round_trip_cost),
        mode=aux_mode_norm,
    )
    y_train = target_all.iloc[train_idx].reset_index(drop=True)
    model = LGBMRegressor(
        objective="regression",
        n_estimators=220,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=70,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=5.0,
        random_state=int(seed),
        n_jobs=2,
        verbosity=-1,
    )
    weights = np.where(train_labels.iloc[train_idx]["clean"].to_numpy(dtype=bool), 1.75, 1.0).astype(np.float32)
    model.fit(x_train, y_train, sample_weight=weights)
    score[valid_idx] = model.predict(fold["x_valid"].iloc[valid_idx].reset_index(drop=True)).astype(np.float32)
    return score


def run_overlay(
    *,
    labels_path: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    best_config_path: Path,
    output_dir: Path,
    months: list[str],
    alphas: list[float],
    max_train_rows: int,
    max_aux_train_rows: int,
    round_trip_cost: float,
    seed: int,
    aux_mode: str,
    include_ae_gmm_state_features: bool,
    ae_gmm_state_feature_max_train_rows: int,
    ae_gmm_state_feature_max_iter: int,
    ae_gmm_state_feature_seed: int,
    ae_gmm_fold_cache_dir: Path | None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _load_config(best_config_path)
    folds, manifest = _prepare_folds(
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
        ae_gmm_state_feature_seed=int(ae_gmm_state_feature_seed),
        ae_gmm_fold_cache_dir=ae_gmm_fold_cache_dir,
        seed=int(seed),
    )
    fold_rows: list[dict[str, Any]] = []
    ledger_parts: list[pd.DataFrame] = []
    for fold_i, fold in enumerate(folds):
        train_label_full = _materialized_soft_label(fold["train_frame"], fold["train_metrics"])
        valid_label = _materialized_soft_label(fold["valid_frame"], fold["valid_metrics"])
        idx = _cap_indices(int(fold["train_rows"]), int(max_train_rows), seed=int(seed) + fold_i * 17)
        x_train = fold["x_train"].iloc[idx].reset_index(drop=True)
        train_frame = fold["train_frame"].iloc[idx].reset_index(drop=True)
        train_metrics = fold["train_metrics"].iloc[idx].reset_index(drop=True)
        train_label = train_label_full.iloc[idx].reset_index(drop=True)
        weights = _ranker_sample_weight(
            train_metrics,
            train_label,
            round_trip_cost=float(round_trip_cost),
            mode="base",
        )
        base_score = _fit_ranker(
            x_train,
            train_frame,
            train_metrics,
            train_label,
            weights,
            fold["x_valid"],
            group_mode="timestamp",
            relevance_mode="cleangross",
            round_trip_cost=float(round_trip_cost),
            seed=int(seed) + fold_i,
        )
        aux_score = _fit_long_aux_score(
            fold=fold,
            round_trip_cost=float(round_trip_cost),
            max_aux_train_rows=int(max_aux_train_rows),
            seed=int(seed) + 1009 * (fold_i + 1),
            aux_mode=str(aux_mode),
        )
        long_mask = (
            pd.to_numeric(fold["valid_metrics"].get("side", pd.Series(1.0, index=fold["valid_metrics"].index)), errors="coerce")
            .fillna(1.0)
            .ge(0.0)
            .to_numpy(dtype=bool)
        )
        base_rank = _rank_pct(base_score)
        aux_rank = np.full(len(base_rank), 0.5, dtype=np.float32)
        if bool(long_mask.any()):
            aux_rank[long_mask] = _rank_pct(aux_score[long_mask])
        for alpha in alphas:
            score = base_rank.copy()
            score[long_mask] = score[long_mask] + float(alpha) * (aux_rank[long_mask] - 0.5)
            variant = f"ranker_timestamp_cleangross_long_{str(aux_mode)}_a{float(alpha):.2f}"
            row = _score_fold(
                pd.Series(score),
                valid_label,
                fold["valid_metrics"],
                fold["month"],
                round_trip_cost=float(round_trip_cost),
            )
            row.update(
                {
                    "variant": variant,
                    "stage": variant,
                    "trial_number": 0,
                    "label_name": f"{config.name}_{variant}",
                    "family": config.family,
                    "alpha": float(alpha),
                    "train_rows": int(len(x_train)),
                    "train_rows_uncapped": int(fold["train_rows"]),
                    "valid_rows": int(fold["valid_rows"]),
                    "target_source": "materialized",
                    "ranker_relevance_mode": "cleangross",
                    "aux_head": str(aux_mode),
                }
            )
            fold_rows.append(row)
            ledger_parts.append(
                _scored_ledger(
                    variant=variant,
                    fold=fold,
                    score=score,
                    valid_label=valid_label.reset_index(drop=True),
                )
            )
    summaries: list[dict[str, Any]] = []
    for alpha in alphas:
        variant = f"ranker_timestamp_cleangross_long_{str(aux_mode)}_a{float(alpha):.2f}"
        rows = [row for row in fold_rows if row["variant"] == variant]
        summary = _summarize_trial(
            variant,
            0,
            config,
            rows,
            objective_mode="precision_topk",
        )
        summary["variant"] = variant
        summary["alpha"] = float(alpha)
        summaries.append(summary)
    summary_df = pd.DataFrame(summaries).sort_values("objective", ascending=False).reset_index(drop=True)
    fold_df = pd.DataFrame(fold_rows)
    ledger_df = pd.concat(ledger_parts, ignore_index=True) if ledger_parts else pd.DataFrame()
    paths = {
        "summary": output_dir / "s52_long_clean_aux_overlay_summary.csv",
        "folds": output_dir / "s52_long_clean_aux_overlay_folds.csv",
        "ledger": output_dir / "s52_long_clean_aux_overlay_scored_ledger.parquet",
        "manifest": output_dir / "manifest.json",
        "report": output_dir / "s52_long_clean_aux_overlay.md",
    }
    summary_df.to_csv(paths["summary"], index=False)
    fold_df.to_csv(paths["folds"], index=False)
    ledger_df.to_parquet(paths["ledger"], index=False)
    manifest.update(
        {
            "scope": "s52_long_clean_aux_overlay",
            "labels_path": str(labels_path),
            "feature_dir": str(feature_dir),
            "feature_list_csv": str(feature_list_csv),
            "best_config_path": str(best_config_path),
            "output_dir": str(output_dir),
            "alphas": [float(v) for v in alphas],
            "max_train_rows": int(max_train_rows),
            "max_aux_train_rows": int(max_aux_train_rows),
            "aux_mode": str(aux_mode),
            "ae_gmm_fold_cache_dir": None if ae_gmm_fold_cache_dir is None else str(ae_gmm_fold_cache_dir),
            "round_trip_cost": float(round_trip_cost),
            "seed": int(seed),
            "model_seed": int(seed),
            "ae_gmm_state_feature_seed": int(ae_gmm_state_feature_seed),
            "config": asdict(config),
            "outputs": {k: str(v) for k, v in paths.items()},
        }
    )
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    _write_report(paths["report"], summary_df, fold_df, manifest)
    return {
        "output_dir": str(output_dir),
        "summary": str(paths["summary"]),
        "folds": str(paths["folds"]),
        "report": str(paths["report"]),
        "top": _json_safe(summary_df.head(5).to_dict(orient="records")),
    }


def _write_report(path: Path, summary: pd.DataFrame, folds: pd.DataFrame, manifest: dict[str, Any]) -> None:
    def fmt(df: pd.DataFrame, cols: list[str]) -> str:
        if df.empty:
            return "No rows."
        view = df[[c for c in cols if c in df.columns]].copy()
        for col in view.columns:
            if pd.api.types.is_float_dtype(view[col]):
                view[col] = view[col].map(lambda v: f"{float(v):.4f}" if pd.notna(v) else "")
        return view.to_markdown(index=False)

    cols = [
        "variant",
        "alpha",
        "objective",
        "mean_top10_ev_weighted_first_touch_precision",
        "mean_top20_ev_weighted_first_touch_precision",
        "mean_top30_ev_weighted_first_touch_precision",
        "mean_top10_first_touch_bad_mae_1r_rate",
        "mean_top10_mean_underwater_bars_before_mfe",
        "mean_top10_mean_ev",
        "mean_long_top10_ev_weighted_first_touch_precision",
        "mean_short_top10_ev_weighted_first_touch_precision",
    ]
    lines = [
        "# S52 Long Clean Aux Overlay",
        "",
        "Scope: blend a prior-month long clean-vs-dirty auxiliary head into the clean-gross ranker score.",
        "",
        f"Rows: `{manifest.get('rows')}`",
        f"Symbols: `{manifest.get('symbols')}`",
        f"Months: `{', '.join(manifest.get('fold_months', []))}`",
        f"Aux mode: `{manifest.get('aux_mode')}`",
        "",
        "## Summary",
        "",
        fmt(summary, cols),
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--best-config-path", type=Path, default=DEFAULT_BEST_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--months", default=",".join(DEFAULT_MONTHS))
    parser.add_argument("--alphas", default="0.0,0.10,0.25,0.50,0.75")
    parser.add_argument(
        "--aux-mode",
        choices=("clean_prob", "clean_gross_rank", "clean_gross_margin"),
        default="clean_prob",
    )
    parser.add_argument("--max-train-rows", type=int, default=150_000)
    parser.add_argument("--max-aux-train-rows", type=int, default=150_000)
    parser.add_argument("--round-trip-cost", type=float, default=DEFAULT_ROUND_TRIP_COST)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-ae-gmm-state-features", action="store_true")
    parser.add_argument("--ae-gmm-state-feature-max-train-rows", type=int, default=60_000)
    parser.add_argument("--ae-gmm-state-feature-max-iter", type=int, default=32)
    parser.add_argument("--ae-gmm-state-feature-seed", type=int, default=42)
    parser.add_argument("--ae-gmm-fold-cache-dir", type=Path, default=None)
    args = parser.parse_args()
    result = run_overlay(
        labels_path=args.labels_path,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        best_config_path=args.best_config_path,
        output_dir=args.output_dir,
        months=_parse_csv(args.months, DEFAULT_MONTHS),
        alphas=_parse_float_csv(args.alphas, (0.0, 0.10, 0.25, 0.50, 0.75)),
        max_train_rows=int(args.max_train_rows),
        max_aux_train_rows=int(args.max_aux_train_rows),
        round_trip_cost=float(args.round_trip_cost),
        seed=int(args.seed),
        aux_mode=str(args.aux_mode),
        include_ae_gmm_state_features=not bool(args.no_ae_gmm_state_features),
        ae_gmm_state_feature_max_train_rows=int(args.ae_gmm_state_feature_max_train_rows),
        ae_gmm_state_feature_max_iter=int(args.ae_gmm_state_feature_max_iter),
        ae_gmm_state_feature_seed=int(args.ae_gmm_state_feature_seed),
        ae_gmm_fold_cache_dir=args.ae_gmm_fold_cache_dir,
    )
    print(json.dumps(_json_safe(result), indent=2))


if __name__ == "__main__":
    main()
