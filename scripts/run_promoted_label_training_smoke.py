#!/usr/bin/env python3
"""Cheap month-forward training smoke for promoted proxy label candidates.

This is not the full production training pipeline. It uses the existing label
artifact feature columns, trains a small ExtraTrees regressor month-forward, and
checks whether a candidate soft label/weight recipe improves ranking of
execution utility versus the vanilla policy-net target.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

from extreme_price_movements.label_weight_optuna import (
    apply_weight_recipe,
    build_native_mfe_mae_soft_label_from_frame,
)


DEFAULT_LABELS_DIR = Path(
    "data_perp/artifacts/"
    "20260701_193000_single_head_monthly_walkforward_forwardburnin_no_window_hpo_no_regime_fe_labels_s10_policy_net/"
    "labels"
)
DEFAULT_RECIPE = Path("docs/promoted_s16_w12_label_weight_recipe.json")
DEFAULT_OUTPUT_DIR = Path("data_perp/reports/promoted_label_training_smoke_v1")
TOP_FRACS = (0.30, 0.10, 0.05, 0.01)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _safe_numeric(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _safe_mean(values: Any) -> float:
    arr = _safe_numeric(values).dropna()
    return float(arr.mean()) if len(arr) else float("nan")


def _safe_quantile(values: Any, q: float) -> float:
    arr = _safe_numeric(values).dropna()
    return float(arr.quantile(q)) if len(arr) else float("nan")


def _spearman(a: Any, b: Any) -> float:
    aa = _safe_numeric(a)
    bb = _safe_numeric(b)
    mask = aa.notna() & bb.notna()
    if int(mask.sum()) < 3:
        return float("nan")
    return float(aa[mask].rank(method="average").corr(bb[mask].rank(method="average")))


def _read_first_dataset(labels_dir: Path) -> tuple[str, pd.DataFrame]:
    manifest_path = labels_dir / "labels_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    datasets = manifest.get("datasets") or {}
    if not isinstance(datasets, dict) or not datasets:
        raise RuntimeError(f"No datasets in {manifest_path}")
    for dataset_name, meta in datasets.items():
        file_name = str((meta or {}).get("file") or "")
        if file_name:
            return str(dataset_name), pd.read_parquet(labels_dir / file_name)
    raise RuntimeError(f"No readable dataset file in {manifest_path}")


def _feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    excluded = {
        "__y_bin__",
        "__y_ret__",
        "__w__",
        "__ts__",
        "__symbol__",
        "__u_policy_net__",
        "__r_policy_net__",
    }
    cols = [
        col
        for col in df.columns
        if col not in excluded and not str(col).startswith("__")
    ]
    x = df[cols].select_dtypes(include=[np.number]).copy()
    if x.empty:
        raise RuntimeError("No numeric feature columns after excluding label/internal columns")
    med = x.median(numeric_only=True)
    x = x.replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
    return x.astype(np.float32, copy=False)


def _build_arm_targets(
    df: pd.DataFrame,
    *,
    arm: str,
    recipe_path: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    y_hard = _safe_numeric(df["__y_bin__"]).fillna(0.0).to_numpy(dtype=np.float32)
    w0 = np.sqrt(
        np.clip(
            _safe_numeric(df["__w__"]).fillna(1.0).to_numpy(dtype=np.float32),
            0.0,
            None,
        )
    )
    if arm == "vanilla_s10":
        cfg = {
            "label_ablation_mode": "policy_net",
            "policy_net_label_center": 0.0,
            "policy_net_label_temperature": 0.004,
            "label_weight_disable": True,
        }
        y_soft, label_stats = build_native_mfe_mae_soft_label_from_frame(
            df,
            y_hard,
            cfg=cfg,
            stage="train_base",
            label=arm,
        )
        return y_soft, w0, {"label_stats": label_stats, "weight_stats": {"enabled": False}}
    if arm == "promoted_s16_w12":
        cfg = {
            "label_ablation_mode": "policy_net",
            "label_weight_base_recipe": str(recipe_path),
        }
        y_soft, label_stats = build_native_mfe_mae_soft_label_from_frame(
            df,
            y_hard,
            cfg=cfg,
            stage="train_base",
            label=arm,
        )
        # Weights are generated per fold with each fold's fit mask.
        return y_soft, w0, {"label_stats": label_stats}
    raise ValueError(f"Unknown arm: {arm}")


def _decile_monotonicity(pred: np.ndarray, utility: pd.Series) -> float:
    frame = pd.DataFrame({"pred": pred, "u": _safe_numeric(utility)})
    frame = frame.dropna()
    if len(frame) < 20:
        return float("nan")
    try:
        frame["decile"] = pd.qcut(
            frame["pred"].rank(method="first"),
            10,
            labels=False,
            duplicates="drop",
        )
    except ValueError:
        return float("nan")
    by_decile = frame.groupby("decile", observed=True)["u"].mean()
    if len(by_decile) < 3:
        return float("nan")
    return _spearman(pd.Series(by_decile.index, dtype=float), by_decile.reset_index(drop=True))


def _selection_metrics(
    *,
    valid: pd.DataFrame,
    pred: np.ndarray,
    arm: str,
    month: str,
    top_frac: float,
) -> dict[str, Any]:
    u = _safe_numeric(valid["__u_policy_net__"])
    n = len(valid)
    k = max(1, int(math.ceil(n * float(top_frac))))
    score = pd.Series(pred, index=valid.index)
    selected = score.rank(method="first", ascending=False) <= k
    u_sel = u[selected]
    return {
        "arm": arm,
        "period": month,
        "top_frac": float(top_frac),
        "rows": int(n),
        "selected_rows": int(selected.sum()),
        "mean_u": _safe_mean(u_sel),
        "hit_u": _safe_mean(u_sel > 0.0),
        "q10_u": _safe_quantile(u_sel, 0.10),
        "period_mean_u": _safe_mean(u),
        "delta_mean_u_vs_period": _safe_mean(u_sel) - _safe_mean(u),
        "score_ic_u": _spearman(score, u),
        "decile_monotonicity_u": _decile_monotonicity(pred, u),
    }


def _fit_predict_month(
    *,
    x: pd.DataFrame,
    df: pd.DataFrame,
    y_soft: np.ndarray,
    base_weight: np.ndarray,
    arm: str,
    recipe_path: Path,
    month: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    month_period = pd.to_datetime(df["__ts__"], errors="coerce").dt.to_period("M").astype(str)
    train_mask = month_period < month
    valid_mask = month_period == month
    if int(train_mask.sum()) < 500 or int(valid_mask.sum()) < 100:
        return [], {"month": month, "skipped": True}
    if arm == "promoted_s16_w12":
        weights, weight_stats = apply_weight_recipe(
            df,
            _safe_numeric(df["__y_bin__"]).fillna(0.0).to_numpy(dtype=np.float32),
            y_soft,
            base_weight,
            cfg={"label_weight_base_recipe": str(recipe_path)},
            stage="train_base",
            label=f"{arm}:{month}",
            fit_mask=train_mask.to_numpy(dtype=bool),
        )
    else:
        weights = base_weight
        weight_stats = {"enabled": False}
    model = ExtraTreesRegressor(
        n_estimators=160,
        max_depth=8,
        min_samples_leaf=32,
        max_features="sqrt",
        random_state=42,
        n_jobs=2,
    )
    model.fit(
        x.loc[train_mask],
        np.asarray(y_soft, dtype=np.float32)[train_mask.to_numpy(dtype=bool)],
        sample_weight=np.asarray(weights, dtype=np.float32)[train_mask.to_numpy(dtype=bool)],
    )
    pred = model.predict(x.loc[valid_mask]).astype(np.float32)
    valid = df.loc[valid_mask].reset_index(drop=True)
    rows = [
        _selection_metrics(
            valid=valid,
            pred=pred,
            arm=arm,
            month=month,
            top_frac=top_frac,
        )
        for top_frac in TOP_FRACS
    ]
    diag = {
        "month": month,
        "arm": arm,
        "train_rows": int(train_mask.sum()),
        "valid_rows": int(valid_mask.sum()),
        "target_train_mean": float(np.mean(np.asarray(y_soft)[train_mask.to_numpy(dtype=bool)])),
        "target_train_std": float(np.std(np.asarray(y_soft)[train_mask.to_numpy(dtype=bool)])),
        "weight_train_mean": float(np.mean(np.asarray(weights)[train_mask.to_numpy(dtype=bool)])),
        "weight_train_p95": float(np.percentile(np.asarray(weights)[train_mask.to_numpy(dtype=bool)], 95)),
        "weight_stats": weight_stats,
    }
    return rows, diag


def _aggregate(monthly: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (arm, top_frac), group in monthly.groupby(["arm", "top_frac"], dropna=False, observed=True):
        mean_u = _safe_numeric(group["mean_u"])
        rows.append(
            {
                "arm": str(arm),
                "top_frac": float(top_frac),
                "months": int(group["period"].nunique()),
                "positive_months": int((mean_u > 0.0).sum()),
                "mean_u": _safe_mean(mean_u),
                "worst_month_mean_u": _safe_quantile(mean_u, 0.0),
                "hit_u": _safe_mean(group["hit_u"]),
                "q10_u": _safe_mean(group["q10_u"]),
                "delta_mean_u_vs_period": _safe_mean(group["delta_mean_u_vs_period"]),
                "score_ic_u": _safe_mean(group["score_ic_u"]),
                "decile_monotonicity_u": _safe_mean(group["decile_monotonicity_u"]),
                "mean_selected_rows": _safe_mean(group["selected_rows"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["top_frac", "mean_u"], ascending=[True, False])


def _table(frame: pd.DataFrame, cols: list[str]) -> str:
    if frame.empty:
        return "No rows."
    view = frame[[col for col in cols if col in frame.columns]].copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda v: f"{float(v):.4f}" if pd.notna(v) else "")
    return view.to_markdown(index=False)


def run_smoke(*, labels_dir: Path, recipe_path: Path, output_dir: Path) -> dict[str, Any]:
    dataset_name, df = _read_first_dataset(labels_dir)
    df = df.reset_index(drop=True)
    df["__ts__"] = pd.to_datetime(df["__ts__"], errors="coerce")
    x = _feature_frame(df)
    months = sorted(df["__ts__"].dt.to_period("M").dropna().astype(str).unique())
    monthly_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    arms = ("vanilla_s10", "promoted_s16_w12")
    for arm in arms:
        y_soft, base_weight, target_diag = _build_arm_targets(df, arm=arm, recipe_path=recipe_path)
        diagnostics.append({"arm": arm, "target": target_diag})
        for month in months[1:]:
            rows, diag = _fit_predict_month(
                x=x,
                df=df,
                y_soft=y_soft,
                base_weight=base_weight,
                arm=arm,
                recipe_path=recipe_path,
                month=month,
            )
            monthly_rows.extend(rows)
            diagnostics.append(diag)
    monthly = pd.DataFrame(monthly_rows)
    aggregate = _aggregate(monthly)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "monthly": output_dir / "training_smoke_monthly.csv",
        "aggregate": output_dir / "training_smoke_aggregate.csv",
        "diagnostics": output_dir / "training_smoke_diagnostics.json",
        "manifest": output_dir / "manifest.json",
        "markdown": output_dir / "promoted_label_training_smoke.md",
    }
    monthly.to_csv(paths["monthly"], index=False)
    aggregate.to_csv(paths["aggregate"], index=False)
    paths["diagnostics"].write_text(json.dumps(_json_safe(diagnostics), indent=2), encoding="utf-8")
    manifest = {
        "scope": "cheap_month_forward_model_smoke_not_full_policy_training",
        "labels_dir": str(labels_dir),
        "dataset": dataset_name,
        "recipe_path": str(recipe_path),
        "output_dir": str(output_dir),
        "rows": int(len(df)),
        "feature_count": int(x.shape[1]),
        "timestamp_min": df["__ts__"].min(),
        "timestamp_max": df["__ts__"].max(),
        "arms": list(arms),
        "top_fracs": list(TOP_FRACS),
        "outputs": {key: str(path) for key, path in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    cols = [
        "arm",
        "top_frac",
        "months",
        "positive_months",
        "mean_u",
        "worst_month_mean_u",
        "hit_u",
        "q10_u",
        "delta_mean_u_vs_period",
        "score_ic_u",
        "decile_monotonicity_u",
        "mean_selected_rows",
    ]
    markdown = [
        "# Promoted Label Training Smoke",
        "",
        "Scope: cheap month-forward model smoke. This is not full policy execution and not a final OOS claim.",
        "",
        "Train windows use only prior months; validation months are Apr-May-Jun 2026 where available.",
        "",
        "## Aggregate",
        "",
        _table(aggregate, cols),
        "",
        "## Monthly",
        "",
        _table(monthly.sort_values(["top_frac", "period", "arm"]), cols + ["period"]),
        "",
        "## Outputs",
        "",
        f"- Monthly: `{paths['monthly']}`",
        f"- Aggregate: `{paths['aggregate']}`",
        f"- Diagnostics: `{paths['diagnostics']}`",
        f"- Manifest: `{paths['manifest']}`",
    ]
    paths["markdown"].write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_smoke(
        labels_dir=args.labels_dir,
        recipe_path=args.recipe,
        output_dir=args.output_dir,
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
