#!/usr/bin/env python3
"""S52 long-side clean-vs-dirty feature separability diagnostic.

This diagnostic checks whether pre-entry features can separate clean long
first-passage opportunities from dirty positive long paths under the current
materialized S52 labels. AUC is reported only as a secondary diagnostic; the
main metrics are top-k clean precision and dirty-path rates.
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

try:
    from lightgbm import LGBMClassifier

    _LIGHTGBM_AVAILABLE = True
except Exception:  # pragma: no cover
    LGBMClassifier = None
    _LIGHTGBM_AVAILABLE = False

from scripts.run_gate3_side_soft_label_hpo import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    DEFAULT_FEATURE_LIST_CSV,
    DEFAULT_ROUND_TRIP_COST,
    _json_safe,
    _prepare_folds,
)
from scripts.run_s52_ranker_smoke import (  # noqa: E402
    DEFAULT_LABELS_PATH,
    DEFAULT_MONTHS,
    _materialized_soft_label,
    _parse_csv,
    _safe_mean,
)


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/s52_long_clean_dirty_separability_20260705_v1")
TOP_FRACS = (0.10, 0.20, 0.30)


def _candidate_labels(
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    *,
    round_trip_cost: float,
    side: str,
) -> pd.DataFrame:
    label = _materialized_soft_label(frame, metrics).reset_index(drop=True)
    side_values = pd.to_numeric(metrics.get("side", pd.Series(1.0, index=metrics.index)), errors="coerce").fillna(1.0)
    if str(side).strip().lower() == "long":
        side_mask = side_values.ge(0.0).reset_index(drop=True)
    elif str(side).strip().lower() == "short":
        side_mask = side_values.lt(0.0).reset_index(drop=True)
    else:
        side_mask = pd.Series(True, index=label.index)
    first_touch_net = pd.to_numeric(
        metrics.get("first_touch_net", metrics.get("u_policy_net", pd.Series(0.0, index=metrics.index))),
        errors="coerce",
    ).fillna(0.0).reset_index(drop=True)
    gross = first_touch_net + float(round_trip_cost)
    clean = pd.to_numeric(label["target_hard"], errors="coerce").fillna(0.0).gt(0.5)
    first_good = pd.to_numeric(label["first_pass_good"], errors="coerce").fillna(0.0).gt(0.5)
    first_bad = pd.to_numeric(label["first_pass_bad"], errors="coerce").fillna(0.0).gt(0.5)
    dirty = pd.to_numeric(label["dirty_positive"], errors="coerce").fillna(0.0).gt(0.5)
    positive = gross.gt(0.0) | pd.to_numeric(label["positive_u"], errors="coerce").fillna(0.0).gt(0.5)
    candidate = side_mask & positive & (clean | dirty | first_good | first_bad)
    y = clean & candidate
    dirty_path = candidate & ~clean & (dirty | first_bad)
    return pd.DataFrame(
        {
            "candidate": candidate.astype(bool),
            "clean": y.astype(bool),
            "dirty": dirty_path.astype(bool),
            "first_good": (first_good & candidate).astype(bool),
            "first_bad": (first_bad & candidate).astype(bool),
            "gross": gross,
            "net": first_touch_net,
        }
    )


def _cap_indices(n_rows: int, max_rows: int, seed: int) -> np.ndarray:
    if int(max_rows) <= 0 or n_rows <= int(max_rows):
        return np.arange(n_rows, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(n_rows, size=int(max_rows), replace=False)
    return np.sort(idx.astype(np.int64))


def _safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.int8)
    s = np.asarray(score, dtype=np.float64)
    mask = np.isfinite(s)
    y = y[mask]
    s = s[mask]
    if len(y) < 2 or len(np.unique(y)) < 2:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    n_pos = float(y.sum())
    n_neg = float(len(y) - y.sum())
    if n_pos <= 0.0 or n_neg <= 0.0:
        return float("nan")
    return float((ranks[y.astype(bool)].sum() - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg))


def _top_metrics(score: np.ndarray, labels: pd.DataFrame, metrics: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    score_s = pd.Series(score).replace([np.inf, -np.inf], np.nan)
    valid = np.flatnonzero(np.isfinite(score_s.to_numpy(dtype=np.float64)))
    order = valid[np.argsort(-score_s.iloc[valid].to_numpy(dtype=np.float64), kind="mergesort")]
    clean = labels["clean"].reset_index(drop=True).astype(bool)
    dirty = labels["dirty"].reset_index(drop=True).astype(bool)
    first_good = labels["first_good"].reset_index(drop=True).astype(bool)
    first_bad = labels["first_bad"].reset_index(drop=True).astype(bool)
    gross = pd.to_numeric(labels["gross"], errors="coerce").reset_index(drop=True)
    net = pd.to_numeric(labels["net"], errors="coerce").reset_index(drop=True)
    ft_mae = pd.to_numeric(metrics.get("first_touch_mae_norm", pd.Series(np.nan, index=metrics.index)), errors="coerce").reset_index(drop=True)
    mae_before = pd.to_numeric(metrics.get("mae_1r_before_mfe_1r", pd.Series(np.nan, index=metrics.index)), errors="coerce").reset_index(drop=True)
    adv = pd.to_numeric(metrics.get("max_adverse_before_mfe_1r", pd.Series(np.nan, index=metrics.index)), errors="coerce").reset_index(drop=True)
    underwater = pd.to_numeric(metrics.get("underwater_bars_before_mfe_1r", pd.Series(np.nan, index=metrics.index)), errors="coerce").reset_index(drop=True)
    for frac in TOP_FRACS:
        tag = f"top{int(round(frac * 100)):02d}"
        k = max(1, int(math.ceil(float(frac) * len(order)))) if len(order) else 0
        idx = order[:k]
        out[f"{tag}_rows"] = int(k)
        out[f"{tag}_clean_precision"] = _safe_mean(clean.iloc[idx]) if k else float("nan")
        out[f"{tag}_dirty_rate"] = _safe_mean(dirty.iloc[idx]) if k else float("nan")
        out[f"{tag}_first_good_rate"] = _safe_mean(first_good.iloc[idx]) if k else float("nan")
        out[f"{tag}_first_bad_rate"] = _safe_mean(first_bad.iloc[idx]) if k else float("nan")
        out[f"{tag}_mean_gross"] = _safe_mean(gross.iloc[idx]) if k else float("nan")
        out[f"{tag}_mean_net"] = _safe_mean(net.iloc[idx]) if k else float("nan")
        out[f"{tag}_first_touch_bad_mae_1r_rate"] = _safe_mean(ft_mae.iloc[idx].ge(1.0)) if k else float("nan")
        out[f"{tag}_mae_1r_before_mfe_1r_rate"] = _safe_mean(mae_before.iloc[idx].gt(0.5)) if k else float("nan")
        out[f"{tag}_mean_max_adverse_before_mfe_1r"] = _safe_mean(adv.iloc[idx]) if k else float("nan")
        out[f"{tag}_mean_underwater_bars_before_mfe_1r"] = _safe_mean(underwater.iloc[idx]) if k else float("nan")
    return out


def _fit_score_fold(
    *,
    fold: dict[str, Any],
    side: str,
    max_train_rows: int,
    round_trip_cost: float,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if not _LIGHTGBM_AVAILABLE or LGBMClassifier is None:
        raise RuntimeError("lightgbm is required")
    train_labels = _candidate_labels(
        fold["train_frame"],
        fold["train_metrics"],
        round_trip_cost=round_trip_cost,
        side=side,
    )
    valid_labels_all = _candidate_labels(
        fold["valid_frame"],
        fold["valid_metrics"],
        round_trip_cost=round_trip_cost,
        side=side,
    )
    train_idx_all = np.flatnonzero(train_labels["candidate"].to_numpy(dtype=bool))
    valid_idx = np.flatnonzero(valid_labels_all["candidate"].to_numpy(dtype=bool))
    train_idx = train_idx_all[_cap_indices(len(train_idx_all), int(max_train_rows), seed=int(seed))]
    row: dict[str, Any] = {
        "month": str(fold["month"]),
        "side": str(side),
        "train_candidates": int(len(train_idx_all)),
        "train_used": int(len(train_idx)),
        "valid_candidates": int(len(valid_idx)),
        "train_clean_rate": _safe_mean(train_labels.iloc[train_idx_all]["clean"]) if len(train_idx_all) else float("nan"),
        "valid_clean_rate": _safe_mean(valid_labels_all.iloc[valid_idx]["clean"]) if len(valid_idx) else float("nan"),
        "valid_dirty_rate": _safe_mean(valid_labels_all.iloc[valid_idx]["dirty"]) if len(valid_idx) else float("nan"),
    }
    if len(train_idx) < 500 or len(valid_idx) < 100 or train_labels.iloc[train_idx]["clean"].nunique() < 2:
        row["status"] = "insufficient_rows"
        return row, pd.DataFrame()

    x_train = fold["x_train"].iloc[train_idx].reset_index(drop=True)
    y_train = train_labels.iloc[train_idx]["clean"].astype(int).reset_index(drop=True)
    x_valid = fold["x_valid"].iloc[valid_idx].reset_index(drop=True)
    valid_labels = valid_labels_all.iloc[valid_idx].reset_index(drop=True)
    valid_metrics = fold["valid_metrics"].iloc[valid_idx].reset_index(drop=True)
    weight = np.where(y_train.to_numpy(dtype=bool), 1.75, 1.0).astype(np.float32)
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
    model.fit(x_train, y_train, sample_weight=weight)
    score = model.predict_proba(x_valid)[:, 1].astype(np.float32)
    y_valid = valid_labels["clean"].astype(int).to_numpy()
    row.update(
        {
            "status": "ok",
            "auc_secondary": _safe_auc(y_valid, score),
            **_top_metrics(score, valid_labels, valid_metrics),
        }
    )
    importance = pd.DataFrame(
        {
            "month": str(fold["month"]),
            "side": str(side),
            "feature": list(x_train.columns),
            "importance": model.booster_.feature_importance(importance_type="gain"),
        }
    )
    return row, importance


def run_diagnostic(
    *,
    labels_path: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    output_dir: Path,
    months: list[str],
    side: str,
    max_train_rows: int,
    round_trip_cost: float,
    seed: int,
    include_ae_gmm_state_features: bool,
    ae_gmm_state_feature_max_train_rows: int,
    ae_gmm_state_feature_max_iter: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
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
        seed=int(seed),
    )
    rows: list[dict[str, Any]] = []
    importances: list[pd.DataFrame] = []
    for i, fold in enumerate(folds):
        row, imp = _fit_score_fold(
            fold=fold,
            side=side,
            max_train_rows=int(max_train_rows),
            round_trip_cost=float(round_trip_cost),
            seed=int(seed) + i * 101,
        )
        rows.append(row)
        if not imp.empty:
            importances.append(imp)
    fold_df = pd.DataFrame(rows)
    imp_df = pd.concat(importances, ignore_index=True) if importances else pd.DataFrame()
    if not imp_df.empty:
        imp_summary = (
            imp_df.groupby("feature", observed=True)["importance"]
            .agg(["mean", "sum", "count"])
            .reset_index()
            .sort_values("mean", ascending=False)
        )
    else:
        imp_summary = pd.DataFrame()
    summary = {
        "side": str(side),
        "folds": int(len(fold_df)),
        "ok_folds": int(fold_df["status"].eq("ok").sum()) if "status" in fold_df.columns else 0,
        "mean_valid_clean_rate": _safe_mean(fold_df.get("valid_clean_rate", [])),
        "mean_valid_dirty_rate": _safe_mean(fold_df.get("valid_dirty_rate", [])),
        "mean_auc_secondary": _safe_mean(fold_df.get("auc_secondary", [])),
        "mean_top10_clean_precision": _safe_mean(fold_df.get("top10_clean_precision", [])),
        "mean_top20_clean_precision": _safe_mean(fold_df.get("top20_clean_precision", [])),
        "mean_top30_clean_precision": _safe_mean(fold_df.get("top30_clean_precision", [])),
        "mean_top10_dirty_rate": _safe_mean(fold_df.get("top10_dirty_rate", [])),
        "mean_top10_first_touch_bad_mae_1r_rate": _safe_mean(fold_df.get("top10_first_touch_bad_mae_1r_rate", [])),
        "mean_top10_mae_1r_before_mfe_1r_rate": _safe_mean(fold_df.get("top10_mae_1r_before_mfe_1r_rate", [])),
        "mean_top10_mean_underwater_bars_before_mfe_1r": _safe_mean(fold_df.get("top10_mean_underwater_bars_before_mfe_1r", [])),
    }
    paths = {
        "summary": output_dir / "s52_long_clean_dirty_separability_summary.json",
        "folds": output_dir / "s52_long_clean_dirty_separability_folds.csv",
        "feature_importance": output_dir / "s52_long_clean_dirty_feature_importance.csv",
        "manifest": output_dir / "manifest.json",
        "report": output_dir / "s52_long_clean_dirty_separability.md",
    }
    paths["summary"].write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    fold_df.to_csv(paths["folds"], index=False)
    imp_summary.to_csv(paths["feature_importance"], index=False)
    manifest.update(
        {
            "scope": "s52_long_clean_dirty_feature_separability",
            "labels_path": str(labels_path),
            "feature_dir": str(feature_dir),
            "feature_list_csv": str(feature_list_csv),
            "output_dir": str(output_dir),
            "side": str(side),
            "max_train_rows": int(max_train_rows),
            "round_trip_cost": float(round_trip_cost),
            "seed": int(seed),
            "include_ae_gmm_state_features": bool(include_ae_gmm_state_features),
            "outputs": {k: str(v) for k, v in paths.items()},
        }
    )
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    _write_report(paths["report"], summary, fold_df, imp_summary, manifest)
    return {"summary": summary, "outputs": {k: str(v) for k, v in paths.items()}}


def _write_report(
    path: Path,
    summary: dict[str, Any],
    folds: pd.DataFrame,
    importance: pd.DataFrame,
    manifest: dict[str, Any],
) -> None:
    def fmt(df: pd.DataFrame, cols: list[str], limit: int | None = None) -> str:
        if df.empty:
            return "No rows."
        view = df[[c for c in cols if c in df.columns]].copy()
        if limit is not None:
            view = view.head(limit)
        for col in view.columns:
            if pd.api.types.is_float_dtype(view[col]):
                view[col] = view[col].map(lambda v: f"{float(v):.4f}" if pd.notna(v) else "")
        return view.to_markdown(index=False)

    lines = [
        "# S52 Long Clean-Vs-Dirty Separability",
        "",
        "Scope: OOS prior-month classifier over current pre-entry features. AUC is secondary; top-k clean precision is the decision metric.",
        "",
        f"Rows: `{manifest.get('rows')}`",
        f"Symbols: `{manifest.get('symbols')}`",
        f"Months: `{', '.join(manifest.get('fold_months', []))}`",
        f"Side: `{manifest.get('side')}`",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(_json_safe(summary), indent=2),
        "```",
        "",
        "## Fold Metrics",
        "",
        fmt(
            folds,
            [
                "month",
                "status",
                "valid_candidates",
                "valid_clean_rate",
                "valid_dirty_rate",
                "auc_secondary",
                "top10_clean_precision",
                "top20_clean_precision",
                "top30_clean_precision",
                "top10_dirty_rate",
                "top10_first_touch_bad_mae_1r_rate",
                "top10_mae_1r_before_mfe_1r_rate",
                "top10_mean_underwater_bars_before_mfe_1r",
            ],
        ),
        "",
        "## Top Features",
        "",
        fmt(importance, ["feature", "mean", "sum", "count"], limit=40),
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--months", default=",".join(DEFAULT_MONTHS))
    parser.add_argument("--side", choices=("long", "short", "all"), default="long")
    parser.add_argument("--max-train-rows", type=int, default=150_000)
    parser.add_argument("--round-trip-cost", type=float, default=DEFAULT_ROUND_TRIP_COST)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-ae-gmm-state-features", action="store_true")
    parser.add_argument("--ae-gmm-state-feature-max-train-rows", type=int, default=60_000)
    parser.add_argument("--ae-gmm-state-feature-max-iter", type=int, default=32)
    args = parser.parse_args()
    result = run_diagnostic(
        labels_path=args.labels_path,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        output_dir=args.output_dir,
        months=_parse_csv(args.months, DEFAULT_MONTHS),
        side=str(args.side),
        max_train_rows=int(args.max_train_rows),
        round_trip_cost=float(args.round_trip_cost),
        seed=int(args.seed),
        include_ae_gmm_state_features=not bool(args.no_ae_gmm_state_features),
        ae_gmm_state_feature_max_train_rows=int(args.ae_gmm_state_feature_max_train_rows),
        ae_gmm_state_feature_max_iter=int(args.ae_gmm_state_feature_max_iter),
    )
    print(json.dumps(_json_safe(result), indent=2))


if __name__ == "__main__":
    main()
