#!/usr/bin/env python3
"""Report base/meta ranking and vol-normalized path-hit metrics for monthly folds."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


DEFAULT_EXPERIMENT_ID = (
    "20260701_193000_single_head_monthly_walkforward_forwardburnin_"
    "no_window_hpo_no_regime_fe"
)

FOLD_MONTHS = {
    "train_march_score_april": {
        "eval_month": "2026-04",
        "validation_start": pd.Timestamp("2026-04-16", tz="UTC"),
        "validation_end": pd.Timestamp("2026-05-01", tz="UTC"),
        "full_start": pd.Timestamp("2026-04-01", tz="UTC"),
        "full_end": pd.Timestamp("2026-05-01", tz="UTC"),
    },
    "train_april_score_may": {
        "eval_month": "2026-05",
        "validation_start": pd.Timestamp("2026-05-16", tz="UTC"),
        "validation_end": pd.Timestamp("2026-06-01", tz="UTC"),
        "full_start": pd.Timestamp("2026-05-01", tz="UTC"),
        "full_end": pd.Timestamp("2026-06-01", tz="UTC"),
    },
    "train_may_score_june": {
        "eval_month": "2026-06",
        "validation_start": pd.Timestamp("2026-06-16", tz="UTC"),
        "validation_end": pd.Timestamp("2026-07-01", tz="UTC"),
        "full_start": pd.Timestamp("2026-06-01", tz="UTC"),
        "full_end": pd.Timestamp("2026-07-01", tz="UTC"),
    },
}


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _safe_mean(values: pd.Series | np.ndarray) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(arr.mean()) if len(arr) else float("nan")


def _safe_spearman(x: Any, y: Any) -> float:
    x_arr = pd.to_numeric(pd.Series(x), errors="coerce")
    y_arr = pd.to_numeric(pd.Series(y), errors="coerce")
    mask = x_arr.notna() & y_arr.notna()
    if int(mask.sum()) < 3:
        return float("nan")
    if x_arr[mask].nunique(dropna=True) < 2 or y_arr[mask].nunique(dropna=True) < 2:
        return float("nan")
    return float(spearmanr(x_arr[mask], y_arr[mask]).correlation)


def _safe_auc(y: Any, score: Any) -> float:
    y_arr = pd.to_numeric(pd.Series(y), errors="coerce")
    s_arr = pd.to_numeric(pd.Series(score), errors="coerce")
    mask = y_arr.notna() & s_arr.notna()
    if int(mask.sum()) < 3:
        return float("nan")
    y_bin = (y_arr[mask].to_numpy(dtype=float) >= 0.5).astype(int)
    if len(np.unique(y_bin)) < 2 or s_arr[mask].nunique(dropna=True) < 2:
        return float("nan")
    return float(roc_auc_score(y_bin, s_arr[mask].to_numpy(dtype=float)))


def _safe_ap(y: Any, score: Any) -> float:
    y_arr = pd.to_numeric(pd.Series(y), errors="coerce")
    s_arr = pd.to_numeric(pd.Series(score), errors="coerce")
    mask = y_arr.notna() & s_arr.notna()
    if int(mask.sum()) < 3:
        return float("nan")
    y_bin = (y_arr[mask].to_numpy(dtype=float) >= 0.5).astype(int)
    if len(np.unique(y_bin)) < 2 or s_arr[mask].nunique(dropna=True) < 2:
        return float("nan")
    return float(average_precision_score(y_bin, s_arr[mask].to_numpy(dtype=float)))


def _score_order_mask(score: pd.Series, frac: float) -> np.ndarray:
    s = pd.to_numeric(score, errors="coerce")
    valid = s.notna().to_numpy()
    if int(valid.sum()) == 0:
        return np.zeros(len(s), dtype=bool)
    if s[valid].nunique(dropna=True) < 2:
        return valid
    valid_idx = np.flatnonzero(valid)
    k = max(1, int(math.ceil(float(frac) * len(valid_idx))))
    ordered = valid_idx[np.argsort(s.iloc[valid_idx].to_numpy(dtype=float), kind="mergesort")]
    out = np.zeros(len(s), dtype=bool)
    out[ordered[-k:]] = True
    return out


def _volnorm_hits(frame: pd.DataFrame) -> dict[str, pd.Series]:
    mfe = pd.to_numeric(frame.get("mfe_ret"), errors="coerce")
    mae = pd.to_numeric(frame.get("mae_ret"), errors="coerce").abs()
    barrier = pd.to_numeric(frame.get("barrier_pct"), errors="coerce").abs()
    valid = barrier > 0.0
    mfe_norm = pd.Series(np.nan, index=frame.index, dtype=float)
    mae_norm = pd.Series(np.nan, index=frame.index, dtype=float)
    mfe_norm.loc[valid] = mfe.loc[valid] / barrier.loc[valid]
    mae_norm.loc[valid] = mae.loc[valid] / barrier.loc[valid]
    return {
        "mfe_norm": mfe_norm,
        "mae_norm": mae_norm,
        "hit_vn_2_1": ((mfe_norm >= 2.0) & (mae_norm < 1.0)).astype(float),
        "hit_vn_3_2": ((mfe_norm >= 3.0) & (mae_norm < 2.0)).astype(float),
    }


def _decile_frame(
    frame: pd.DataFrame,
    *,
    score_col: str,
    month: str,
    layer: str,
    sample: str,
) -> pd.DataFrame:
    score = pd.to_numeric(frame[score_col], errors="coerce")
    valid = score.notna()
    if int(valid.sum()) == 0:
        return pd.DataFrame()
    work = frame.loc[valid].copy()
    score_v = score.loc[valid]
    if score_v.nunique(dropna=True) < 2:
        work["score_decile"] = 10
    else:
        pct_rank = score_v.rank(method="first", pct=True)
        work["score_decile"] = np.clip(np.ceil(pct_rank * 10.0), 1, 10).astype(int)
    hits = _volnorm_hits(work)
    for key, val in hits.items():
        work[key] = val
    rows: list[dict[str, Any]] = []
    for decile, group in work.groupby("score_decile", observed=True):
        y = pd.to_numeric(group.get("y_bin"), errors="coerce")
        ret = pd.to_numeric(group.get("return", group.get("y_ret")), errors="coerce")
        rows.append(
            {
                "eval_month": month,
                "layer": layer,
                "sample": sample,
                "score_decile": int(decile),
                "rows": int(len(group)),
                "score_mean": _safe_mean(group[score_col]),
                "y_bin_hit_rate": _safe_mean(y),
                "mean_return": _safe_mean(ret),
                "hr_vn_2_1": _safe_mean(group["hit_vn_2_1"]),
                "hr_vn_3_2": _safe_mean(group["hit_vn_3_2"]),
                "mfe_norm_mean": _safe_mean(group["mfe_norm"]),
                "mae_norm_mean": _safe_mean(group["mae_norm"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["score_decile"]).reset_index(drop=True)


def _metric_row(
    frame: pd.DataFrame,
    *,
    score_col: str,
    month: str,
    layer: str,
    sample: str,
    run_id: str,
) -> dict[str, Any]:
    if score_col not in frame.columns:
        return {
            "eval_month": month,
            "layer": layer,
            "sample": sample,
            "run_id": run_id,
            "score_col": score_col,
            "rows": 0,
            "reason": "missing_score_col",
        }
    work = frame.copy()
    score = pd.to_numeric(work[score_col], errors="coerce")
    valid = score.notna()
    if "y_bin" in work.columns:
        valid &= pd.to_numeric(work["y_bin"], errors="coerce").notna()
    if int(valid.sum()) == 0:
        return {
            "eval_month": month,
            "layer": layer,
            "sample": sample,
            "run_id": run_id,
            "score_col": score_col,
            "rows": 0,
            "reason": "no_valid_rows",
        }
    work = work.loc[valid].copy()
    score = pd.to_numeric(work[score_col], errors="coerce")
    y = pd.to_numeric(work.get("y_bin"), errors="coerce")
    ret = pd.to_numeric(work.get("return", work.get("y_ret")), errors="coerce")
    hits = _volnorm_hits(work)
    for key, val in hits.items():
        work[key] = val
    base_rate = _safe_mean(y)
    row: dict[str, Any] = {
        "eval_month": month,
        "layer": layer,
        "sample": sample,
        "run_id": run_id,
        "score_col": score_col,
        "rows": int(len(work)),
        "score_unique": int(score.nunique(dropna=True)),
        "score_std": float(score.std(ddof=0)) if len(score.dropna()) else float("nan"),
        "rank_degenerate": bool(score.nunique(dropna=True) < 2),
        "base_hit_rate": base_rate,
        "base_mean_return": _safe_mean(ret),
        "auc": _safe_auc(y, score),
        "pr_auc": _safe_ap(y, score),
        "ic_return": _safe_spearman(score, ret),
        "ic_y_bin": _safe_spearman(score, y),
        "vn_2_1_base_hr": _safe_mean(work["hit_vn_2_1"]),
        "vn_3_2_base_hr": _safe_mean(work["hit_vn_3_2"]),
        "mfe_norm_mean": _safe_mean(work["mfe_norm"]),
        "mae_norm_mean": _safe_mean(work["mae_norm"]),
    }
    for frac in (0.30, 0.10):
        tag = str(int(round(frac * 100)))
        top_mask = _score_order_mask(score, frac)
        top = work.loc[top_mask].copy()
        top_y = pd.to_numeric(top.get("y_bin"), errors="coerce")
        top_ret = pd.to_numeric(top.get("return", top.get("y_ret")), errors="coerce")
        top_hr = _safe_mean(top_y)
        row[f"top{tag}_rows"] = int(len(top))
        row[f"lift_at_{tag}"] = (
            float(top_hr / base_rate)
            if base_rate is not None and np.isfinite(base_rate) and base_rate > 0
            else float("nan")
        )
        row[f"hit_rate_at_{tag}"] = top_hr
        row[f"mean_return_at_{tag}"] = _safe_mean(top_ret)
        for spec in ("vn_2_1", "vn_3_2"):
            hr = _safe_mean(top[f"hit_{spec}"])
            base = row[f"{spec}_base_hr"]
            row[f"hr_{spec}_at_{tag}"] = hr
            row[f"lift_{spec}_at_{tag}"] = (
                float(hr / base)
                if base is not None and np.isfinite(base) and base > 0
                else float("nan")
            )
    deciles = _decile_frame(work, score_col=score_col, month=month, layer=layer, sample=sample)
    if deciles.empty or deciles["score_decile"].nunique() < 2:
        row["decile_count"] = int(deciles["score_decile"].nunique()) if not deciles.empty else 0
        row["decile_spearman_y_bin"] = float("nan")
        row["decile_spearman_return"] = float("nan")
        row["decile_top_bottom_hit_spread"] = float("nan")
        row["decile_top_bottom_return_spread"] = float("nan")
    else:
        row["decile_count"] = int(deciles["score_decile"].nunique())
        row["decile_spearman_y_bin"] = _safe_spearman(
            deciles["score_decile"], deciles["y_bin_hit_rate"]
        )
        row["decile_spearman_return"] = _safe_spearman(
            deciles["score_decile"], deciles["mean_return"]
        )
        bottom = deciles.sort_values("score_decile").iloc[0]
        top = deciles.sort_values("score_decile").iloc[-1]
        row["decile_top_bottom_hit_spread"] = float(top["y_bin_hit_rate"] - bottom["y_bin_hit_rate"])
        row["decile_top_bottom_return_spread"] = float(top["mean_return"] - bottom["mean_return"])
    return row


def _month_from_run_id(run_id: str) -> tuple[str, dict[str, Any]] | None:
    for token, meta in FOLD_MONTHS.items():
        if token in run_id:
            return token, meta
    return None


def _policy_oos_frame(run_root: Path) -> pd.DataFrame:
    path = next((run_root / "policy_oos_predictions").glob("policy_oos_*_clf.parquet"))
    frame = pd.read_parquet(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    return frame


def _base_oof_frame(run_root: Path) -> pd.DataFrame:
    path = next((run_root / "oof").glob("oof_*_H5.parquet"))
    frame = pd.read_parquet(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if "return" not in frame.columns and "y_ret" in frame.columns:
        frame["return"] = pd.to_numeric(frame["y_ret"], errors="coerce")
    return frame


def _meta_oof_frame(run_root: Path) -> pd.DataFrame:
    path = next((run_root / "meta_oof").glob("meta_oof_*_tbm_clf.parquet"))
    frame = pd.read_parquet(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    return frame


def build_report(experiment_id: str, data_root: Path, output_dir: Path) -> dict[str, str]:
    artifact_root = data_root / "artifacts"
    report_root = data_root / "reports" / experiment_id
    runs = sorted(
        p
        for p in artifact_root.glob(f"{experiment_id}_train_*_score_*")
        if p.is_dir() and _month_from_run_id(p.name) is not None
    )
    metric_rows: list[dict[str, Any]] = []
    decile_rows: list[pd.DataFrame] = []
    for run_root in runs:
        resolved = _month_from_run_id(run_root.name)
        if resolved is None:
            continue
        _, meta = resolved
        month = str(meta["eval_month"])
        policy = _policy_oos_frame(run_root)
        validation = policy[
            (policy["timestamp"] >= meta["validation_start"])
            & (policy["timestamp"] < meta["validation_end"])
        ].copy()
        full = policy[
            (policy["timestamp"] >= meta["full_start"])
            & (policy["timestamp"] < meta["full_end"])
        ].copy()
        for sample, frame in (
            ("policy_oos_validation", validation),
            ("policy_oos_full_month", full),
        ):
            for layer, score_col in (
                ("base", "oof_base_clf"),
                ("meta", "clf"),
            ):
                metric_rows.append(
                    _metric_row(
                        frame,
                        score_col=score_col,
                        month=month,
                        layer=layer,
                        sample=sample,
                        run_id=run_root.name,
                    )
                )
                decile_rows.append(
                    _decile_frame(
                        frame,
                        score_col=score_col,
                        month=month,
                        layer=layer,
                        sample=sample,
                    )
                )
        for layer, frame, score_col in (
            ("base", _base_oof_frame(run_root), "oof_prob"),
            ("meta", _meta_oof_frame(run_root), "oof_meta_clf"),
        ):
            metric_rows.append(
                _metric_row(
                    frame,
                    score_col=score_col,
                    month=month,
                    layer=layer,
                    sample="model_oof_context",
                    run_id=run_root.name,
                )
            )
            decile_rows.append(
                _decile_frame(
                    frame,
                    score_col=score_col,
                    month=month,
                    layer=layer,
                    sample="model_oof_context",
                )
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(metric_rows).sort_values(["sample", "eval_month", "layer"])
    deciles = (
        pd.concat([d for d in decile_rows if d is not None and not d.empty], ignore_index=True)
        if decile_rows
        else pd.DataFrame()
    )
    metrics_path = output_dir / "base_meta_walkforward_model_metrics.csv"
    deciles_path = output_dir / "base_meta_walkforward_deciles.csv"
    validation_path = output_dir / "base_meta_policy_oos_validation_metrics.csv"
    oof_path = output_dir / "base_meta_model_oof_context_metrics.csv"
    full_month_path = output_dir / "base_meta_policy_oos_full_month_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    deciles.to_csv(deciles_path, index=False)
    metrics[metrics["sample"].eq("policy_oos_validation")].to_csv(validation_path, index=False)
    metrics[metrics["sample"].eq("model_oof_context")].to_csv(oof_path, index=False)
    metrics[metrics["sample"].eq("policy_oos_full_month")].to_csv(full_month_path, index=False)

    md_path = output_dir / "base_meta_walkforward_model_metrics.md"
    view_cols = [
        "eval_month",
        "layer",
        "sample",
        "rows",
        "score_unique",
        "lift_at_30",
        "lift_at_10",
        "hit_rate_at_30",
        "hit_rate_at_10",
        "hr_vn_2_1_at_30",
        "hr_vn_2_1_at_10",
        "hr_vn_3_2_at_30",
        "hr_vn_3_2_at_10",
        "ic_return",
        "ic_y_bin",
        "decile_spearman_y_bin",
        "decile_spearman_return",
        "auc",
        "pr_auc",
        "base_mean_return",
        "mean_return_at_10",
    ]
    validation_view = metrics[metrics["sample"].eq("policy_oos_validation")][view_cols]
    lines = [
        "# Base/Meta Walk-Forward Model Metrics",
        "",
        "Main table: exact policy-OOS validation windows.",
        "",
        validation_view.to_markdown(index=False),
        "",
        "Definitions: `vn_2_1` means MFE/barrier >= 2 and abs(MAE)/barrier < 1; `vn_3_2` means MFE/barrier >= 3 and abs(MAE)/barrier < 2. Lift columns without `vn` use `y_bin`.",
        "",
        f"Full CSV: `{metrics_path}`",
        f"Deciles CSV: `{deciles_path}`",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "experiment_id": experiment_id,
        "report_root": str(report_root),
        "output_dir": str(output_dir),
        "metrics": str(metrics_path),
        "deciles": str(deciles_path),
        "policy_oos_validation": str(validation_path),
        "model_oof_context": str(oof_path),
        "policy_oos_full_month": str(full_month_path),
        "markdown": str(md_path),
        "runs": [str(p.name) for p in runs],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "metrics": str(metrics_path),
        "deciles": str(deciles_path),
        "policy_oos_validation": str(validation_path),
        "model_oof_context": str(oof_path),
        "policy_oos_full_month": str(full_month_path),
        "markdown": str(md_path),
        "manifest": str(manifest_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
    parser.add_argument("--data-root", type=Path, default=Path("data_perp"))
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = (
        args.output_dir
        or args.data_root
        / "reports"
        / str(args.experiment_id)
        / "base_meta_model_metrics"
    )
    print(json.dumps(build_report(str(args.experiment_id), args.data_root, output_dir), indent=2))


if __name__ == "__main__":
    main()
