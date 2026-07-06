#!/usr/bin/env python3
"""Run diagnostics 1-5, 7, and 9 for the monthly walk-forward artifacts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import report_base_meta_walkforward_model_metrics as model_metrics


DEFAULT_EXPERIMENT_ID = (
    "20260701_193000_single_head_monthly_walkforward_forwardburnin_"
    "no_window_hpo_no_regime_fe"
)

FOLD_MONTHS: dict[str, dict[str, Any]] = {
    "train_march_score_april": {
        "eval_month": "2026-04",
        "validation_start": pd.Timestamp("2026-04-16", tz="UTC"),
        "validation_end": pd.Timestamp("2026-05-01", tz="UTC"),
    },
    "train_april_score_may": {
        "eval_month": "2026-05",
        "validation_start": pd.Timestamp("2026-05-16", tz="UTC"),
        "validation_end": pd.Timestamp("2026-06-01", tz="UTC"),
    },
    "train_may_score_june": {
        "eval_month": "2026-06",
        "validation_start": pd.Timestamp("2026-06-16", tz="UTC"),
        "validation_end": pd.Timestamp("2026-07-01", tz="UTC"),
    },
}

TOP_FRACS = (0.30, 0.15, 0.10)
REGIME_COLS = [
    "realized_volatility_24h",
    "vol_z_30_calm",
    "median_spread_bps",
    "base_lgbm_inference_drift_score",
    "meta_lgbm_inference_drift_score",
    "base_lgbm_uncertainty_score",
    "meta_lgbm_uncertainty_score",
    "market_breadth_1h",
    "btc_ret_48h_pct",
]


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


def _safe_mean(values: Any) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(arr.mean()) if len(arr) else float("nan")


def _safe_sum(values: Any) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(arr.sum()) if len(arr) else float("nan")


def _safe_quantile(values: Any, q: float) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(arr.quantile(q)) if len(arr) else float("nan")


def _safe_spearman(x: Any, y: Any) -> float:
    x_arr = pd.to_numeric(pd.Series(x), errors="coerce")
    y_arr = pd.to_numeric(pd.Series(y), errors="coerce")
    mask = x_arr.notna() & y_arr.notna()
    if int(mask.sum()) < 3:
        return float("nan")
    if x_arr[mask].nunique(dropna=True) < 2 or y_arr[mask].nunique(dropna=True) < 2:
        return float("nan")
    return float(spearmanr(x_arr[mask], y_arr[mask]).correlation)


def _safe_ks(reference: Any, current: Any) -> float:
    ref = pd.to_numeric(pd.Series(reference), errors="coerce").dropna()
    cur = pd.to_numeric(pd.Series(current), errors="coerce").dropna()
    if len(ref) < 20 or len(cur) < 20:
        return float("nan")
    if ref.nunique(dropna=True) < 2 or cur.nunique(dropna=True) < 2:
        return 0.0
    return float(ks_2samp(ref.to_numpy(dtype=float), cur.to_numpy(dtype=float)).statistic)


def _psi(reference: Any, current: Any, bins: int = 10) -> float:
    ref = pd.to_numeric(pd.Series(reference), errors="coerce").dropna().to_numpy(dtype=float)
    cur = pd.to_numeric(pd.Series(current), errors="coerce").dropna().to_numpy(dtype=float)
    if len(ref) < 20 or len(cur) < 20:
        return float("nan")
    if np.nanstd(ref) == 0.0:
        return 0.0 if np.nanstd(cur) == 0.0 else float("inf")
    quantiles = np.unique(np.nanquantile(ref, np.linspace(0.0, 1.0, bins + 1)))
    if len(quantiles) < 3:
        return 0.0
    edges = np.concatenate(([-np.inf], quantiles[1:-1], [np.inf]))
    ref_hist = np.histogram(ref, bins=edges)[0].astype(float) / max(len(ref), 1)
    cur_hist = np.histogram(cur, bins=edges)[0].astype(float) / max(len(cur), 1)
    eps = 1e-6
    return float(np.sum((cur_hist - ref_hist) * np.log((cur_hist + eps) / (ref_hist + eps))))


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        val = float(value)
    except Exception:
        return ""
    if not math.isfinite(val):
        return ""
    return f"{val:.{digits}f}"


def _month_from_run_id(run_id: str) -> tuple[str, dict[str, Any]] | None:
    for token, meta in FOLD_MONTHS.items():
        if token in run_id:
            return token, meta
    return None


def _policy_oos_frame(run_root: Path) -> pd.DataFrame:
    path = next((run_root / "policy_oos_predictions").glob("policy_oos_*_clf.parquet"))
    frame = pd.read_parquet(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    return _with_path_metrics(frame)


def _oof_frame(run_root: Path) -> pd.DataFrame:
    path = next((run_root / "oof").glob("oof_*_H5.parquet"))
    frame = pd.read_parquet(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if "return" not in frame.columns and "y_ret" in frame.columns:
        frame["return"] = pd.to_numeric(frame["y_ret"], errors="coerce")
    return _with_path_metrics(frame)


def _meta_oof_frame(run_root: Path) -> pd.DataFrame:
    path = next((run_root / "meta_oof").glob("meta_oof_*_tbm_clf.parquet"))
    frame = pd.read_parquet(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if "return" not in frame.columns and "y_ret" in frame.columns:
        frame["return"] = pd.to_numeric(frame["y_ret"], errors="coerce")
    return _with_path_metrics(frame)


def _with_path_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    mfe = pd.to_numeric(out.get("mfe_ret"), errors="coerce")
    mae = pd.to_numeric(out.get("mae_ret"), errors="coerce").abs()
    barrier = pd.to_numeric(out.get("barrier_pct"), errors="coerce").abs()
    valid = barrier > 0.0
    out["mfe_norm"] = np.nan
    out["mae_norm"] = np.nan
    out.loc[valid, "mfe_norm"] = mfe.loc[valid] / barrier.loc[valid]
    out.loc[valid, "mae_norm"] = mae.loc[valid] / barrier.loc[valid]
    out["hit_vn_2_1"] = ((out["mfe_norm"] >= 2.0) & (out["mae_norm"] < 1.0)).astype(float)
    out["hit_vn_3_2"] = ((out["mfe_norm"] >= 3.0) & (out["mae_norm"] < 2.0)).astype(float)
    out["mfe_ge_1"] = (out["mfe_norm"] >= 1.0).astype(float)
    out["mfe_ge_2"] = (out["mfe_norm"] >= 2.0).astype(float)
    out["mfe_ge_3"] = (out["mfe_norm"] >= 3.0).astype(float)
    out["mae_ge_1"] = (out["mae_norm"] >= 1.0).astype(float)
    out["mae_ge_2"] = (out["mae_norm"] >= 2.0).astype(float)
    return out


def _top_mask(frame: pd.DataFrame, score_col: str, frac: float) -> tuple[pd.Series, bool]:
    score = pd.to_numeric(frame.get(score_col), errors="coerce")
    valid = score.notna()
    if int(valid.sum()) == 0:
        return pd.Series(False, index=frame.index), True
    if score[valid].nunique(dropna=True) < 2:
        return valid.astype(bool), True
    k = max(1, int(math.ceil(frac * int(valid.sum()))))
    selected_idx = score[valid].sort_values(ascending=False, kind="mergesort").index[:k]
    mask = pd.Series(False, index=frame.index)
    mask.loc[selected_idx] = True
    return mask, False


def _periods(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> Iterable[tuple[str, int, pd.Timestamp, pd.Timestamp, pd.DataFrame]]:
    yield "month", 0, start, end, frame.copy()
    week_start = start
    week_index = 1
    while week_start < end:
        week_end = min(week_start + pd.Timedelta(days=7), end)
        week = frame[(frame["timestamp"] >= week_start) & (frame["timestamp"] < week_end)].copy()
        yield "week", week_index, week_start, week_end, week
        week_start = week_end
        week_index += 1


def _slice_stats(frame: pd.DataFrame, extra: dict[str, Any]) -> dict[str, Any]:
    ret = pd.to_numeric(frame.get("return", frame.get("y_ret")), errors="coerce")
    y = pd.to_numeric(frame.get("y_bin"), errors="coerce")
    y_outcome = pd.to_numeric(frame.get("y_outcome"), errors="coerce")
    rows = {
        **extra,
        "rows": int(len(frame)),
        "symbols": int(frame["symbol"].nunique()) if "symbol" in frame.columns else 0,
        "y_bin_rate": _safe_mean(y),
        "mean_return": _safe_mean(ret),
        "median_return": _safe_quantile(ret, 0.5),
        "p10_return": _safe_quantile(ret, 0.1),
        "p90_return": _safe_quantile(ret, 0.9),
        "sum_return": _safe_sum(ret),
        "positive_return_rate": _safe_mean(ret > 0.0),
        "mfe_norm_mean": _safe_mean(frame.get("mfe_norm")),
        "mae_norm_mean": _safe_mean(frame.get("mae_norm")),
        "mfe_ge_1_rate": _safe_mean(frame.get("mfe_ge_1")),
        "mfe_ge_2_rate": _safe_mean(frame.get("mfe_ge_2")),
        "mfe_ge_3_rate": _safe_mean(frame.get("mfe_ge_3")),
        "mae_ge_1_rate": _safe_mean(frame.get("mae_ge_1")),
        "mae_ge_2_rate": _safe_mean(frame.get("mae_ge_2")),
        "hr_vn_2_1": _safe_mean(frame.get("hit_vn_2_1")),
        "hr_vn_3_2": _safe_mean(frame.get("hit_vn_3_2")),
        "bars_to_mfe_mean": _safe_mean(frame.get("bars_to_mfe")),
        "bars_policy_mean": _safe_mean(frame.get("bars_policy")),
        "tp_outcome_rate": _safe_mean(y_outcome == 2),
        "sl_outcome_rate": _safe_mean(y_outcome == 0),
        "timeout_outcome_rate": _safe_mean(y_outcome == 1),
    }
    return rows


def build_label_economics(runs: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_root in runs:
        _, meta = _month_from_run_id(run_root.name) or (None, None)
        if meta is None:
            continue
        month = meta["eval_month"]
        policy = _policy_oos_frame(run_root)
        validation = policy[
            (policy["timestamp"] >= meta["validation_start"])
            & (policy["timestamp"] < meta["validation_end"])
        ].copy()
        for period_type, week_index, start, end, frame in _periods(
            validation, meta["validation_start"], meta["validation_end"]
        ):
            slices: list[tuple[str, pd.DataFrame, bool]] = [("all_base_gate_top40", frame, False)]
            for score_col, prefix in (("oof_base_clf", "base"), ("clf", "meta")):
                for frac in TOP_FRACS:
                    mask, degenerate = _top_mask(frame, score_col, frac)
                    slices.append((f"{prefix}_top{int(frac * 100)}", frame.loc[mask].copy(), degenerate))
            for slice_name, slice_df, degenerate in slices:
                rows.append(
                    _slice_stats(
                        slice_df,
                        {
                            "eval_month": month,
                            "run_id": run_root.name,
                            "period_type": period_type,
                            "week_index": week_index,
                            "period_start": start.isoformat(),
                            "period_end": end.isoformat(),
                            "slice": slice_name,
                            "score_degenerate": bool(degenerate),
                        },
                    )
                )
    return pd.DataFrame(rows)


def _base_feature_list(run_root: Path, limit: int = 60) -> list[str]:
    path = run_root / "quality_reports" / "base_model_feature_importance.csv"
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    if "used_by_model" in frame.columns:
        frame = frame[frame["used_by_model"].fillna(False).astype(bool)].copy()
    sort_cols = [c for c in ("gain_rank", "split_rank", "selected_feature_position") if c in frame.columns]
    if sort_cols:
        frame = frame.sort_values(sort_cols)
    return [str(v) for v in frame["feature"].dropna().drop_duplicates().head(limit).tolist()]


def _meta_feature_list(run_root: Path, limit: int = 100) -> list[str]:
    path = run_root / "meta_oof" / "meta_feature_contract.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    models = data.get("meta_models", {})
    features: list[str] = []
    for entry in models.values():
        for feature in entry.get("feature_columns", []):
            if feature not in features:
                features.append(str(feature))
    return features[:limit]


def _feature_drift_for_layer(
    *,
    eval_month: str,
    run_root: Path,
    layer: str,
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    usable = [f for f in features if f in reference.columns and f in current.columns]
    for feature in usable:
        ref = pd.to_numeric(reference[feature], errors="coerce")
        cur = pd.to_numeric(current[feature], errors="coerce")
        rows.append(
            {
                "diagnostic_type": "reference_vs_oos_feature",
                "eval_month": eval_month,
                "run_id": run_root.name,
                "layer": layer,
                "feature": feature,
                "reference_rows": int(len(reference)),
                "current_rows": int(len(current)),
                "reference_finite_rate": float(ref.notna().mean()) if len(ref) else float("nan"),
                "current_finite_rate": float(cur.notna().mean()) if len(cur) else float("nan"),
                "finite_rate_delta": (
                    float(cur.notna().mean() - ref.notna().mean()) if len(ref) and len(cur) else float("nan")
                ),
                "reference_mean": _safe_mean(ref),
                "current_mean": _safe_mean(cur),
                "mean_delta": _safe_mean(cur) - _safe_mean(ref),
                "reference_std": float(ref.std(ddof=0)) if ref.notna().any() else float("nan"),
                "current_std": float(cur.std(ddof=0)) if cur.notna().any() else float("nan"),
                "reference_p10": _safe_quantile(ref, 0.1),
                "current_p10": _safe_quantile(cur, 0.1),
                "reference_p50": _safe_quantile(ref, 0.5),
                "current_p50": _safe_quantile(cur, 0.5),
                "reference_p90": _safe_quantile(ref, 0.9),
                "current_p90": _safe_quantile(cur, 0.9),
                "psi": _psi(ref, cur),
                "ks": _safe_ks(ref, cur),
                "reference_ic_return": _safe_spearman(ref, reference.get("return", reference.get("y_ret"))),
                "current_ic_return": _safe_spearman(cur, current.get("return", current.get("y_ret"))),
                "reference_ic_y_bin": _safe_spearman(ref, reference.get("y_bin")),
                "current_ic_y_bin": _safe_spearman(cur, current.get("y_bin")),
            }
        )
    return rows


def _embedded_drift_for_layer(
    *,
    eval_month: str,
    run_root: Path,
    layer: str,
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> list[dict[str, Any]]:
    current_prefix = "base_lgbm_" if layer == "base" else "meta_lgbm_"
    rows: list[dict[str, Any]] = []
    for ref_col in reference.columns:
        if not ref_col.startswith("oof_"):
            continue
        metric = ref_col.removeprefix("oof_")
        if not (
            metric.startswith("feature_drift")
            or metric.startswith("row_drift")
            or metric.startswith("regime_centroid")
            or metric
            in {
                "mahalanobis_mean_shift",
                "frobenius_corr_shift",
                "inference_drift_score",
                "uncertainty_score",
                "rare_leaf_low_support_score",
                "contribution_drift_score",
            }
        ):
            continue
        cur_col = f"{current_prefix}{metric}"
        if cur_col not in current.columns:
            continue
        ref = pd.to_numeric(reference[ref_col], errors="coerce")
        cur = pd.to_numeric(current[cur_col], errors="coerce")
        rows.append(
            {
                "diagnostic_type": "embedded_lgbm_drift_metric",
                "eval_month": eval_month,
                "run_id": run_root.name,
                "layer": layer,
                "feature": metric,
                "reference_column": ref_col,
                "current_column": cur_col,
                "reference_rows": int(len(reference)),
                "current_rows": int(len(current)),
                "reference_finite_rate": float(ref.notna().mean()) if len(ref) else float("nan"),
                "current_finite_rate": float(cur.notna().mean()) if len(cur) else float("nan"),
                "finite_rate_delta": (
                    float(cur.notna().mean() - ref.notna().mean()) if len(ref) and len(cur) else float("nan")
                ),
                "reference_mean": _safe_mean(ref),
                "current_mean": _safe_mean(cur),
                "mean_delta": _safe_mean(cur) - _safe_mean(ref),
                "reference_std": float(ref.std(ddof=0)) if ref.notna().any() else float("nan"),
                "current_std": float(cur.std(ddof=0)) if cur.notna().any() else float("nan"),
                "reference_p10": _safe_quantile(ref, 0.1),
                "current_p10": _safe_quantile(cur, 0.1),
                "reference_p50": _safe_quantile(ref, 0.5),
                "current_p50": _safe_quantile(cur, 0.5),
                "reference_p90": _safe_quantile(ref, 0.9),
                "current_p90": _safe_quantile(cur, 0.9),
                "psi": _psi(ref, cur),
                "ks": _safe_ks(ref, cur),
                "reference_ic_return": _safe_spearman(ref, reference.get("return", reference.get("y_ret"))),
                "current_ic_return": _safe_spearman(cur, current.get("return", current.get("y_ret"))),
                "reference_ic_y_bin": _safe_spearman(ref, reference.get("y_bin")),
                "current_ic_y_bin": _safe_spearman(cur, current.get("y_bin")),
            }
        )
    return rows


def _selected_feature_coverage_for_layer(
    *,
    eval_month: str,
    run_root: Path,
    layer: str,
    current: pd.DataFrame,
    features: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    usable = [f for f in features if f in current.columns]
    for feature in usable:
        cur = pd.to_numeric(current[feature], errors="coerce")
        rows.append(
            {
                "diagnostic_type": "selected_feature_oos_coverage",
                "eval_month": eval_month,
                "run_id": run_root.name,
                "layer": layer,
                "feature": feature,
                "reference_rows": 0,
                "current_rows": int(len(current)),
                "reference_finite_rate": float("nan"),
                "current_finite_rate": float(cur.notna().mean()) if len(cur) else float("nan"),
                "finite_rate_delta": float("nan"),
                "reference_mean": float("nan"),
                "current_mean": _safe_mean(cur),
                "mean_delta": float("nan"),
                "reference_std": float("nan"),
                "current_std": float(cur.std(ddof=0)) if cur.notna().any() else float("nan"),
                "reference_p10": float("nan"),
                "current_p10": _safe_quantile(cur, 0.1),
                "reference_p50": float("nan"),
                "current_p50": _safe_quantile(cur, 0.5),
                "reference_p90": float("nan"),
                "current_p90": _safe_quantile(cur, 0.9),
                "psi": float("nan"),
                "ks": float("nan"),
                "reference_ic_return": float("nan"),
                "current_ic_return": _safe_spearman(cur, current.get("return", current.get("y_ret"))),
                "reference_ic_y_bin": float("nan"),
                "current_ic_y_bin": _safe_spearman(cur, current.get("y_bin")),
            }
        )
    return rows


def build_feature_drift(runs: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_root in runs:
        _, meta = _month_from_run_id(run_root.name) or (None, None)
        if meta is None:
            continue
        month = str(meta["eval_month"])
        policy = _policy_oos_frame(run_root)
        validation = policy[
            (policy["timestamp"] >= meta["validation_start"])
            & (policy["timestamp"] < meta["validation_end"])
        ].copy()
        base_oof = _oof_frame(run_root)
        meta_oof = _meta_oof_frame(run_root)
        base_features = _base_feature_list(run_root)
        meta_features = _meta_feature_list(run_root)
        rows.extend(
            _feature_drift_for_layer(
                eval_month=month,
                run_root=run_root,
                layer="base",
                reference=base_oof,
                current=validation,
                features=base_features,
            )
        )
        rows.extend(
            _embedded_drift_for_layer(
                eval_month=month,
                run_root=run_root,
                layer="base",
                reference=base_oof,
                current=validation,
            )
        )
        rows.extend(
            _selected_feature_coverage_for_layer(
                eval_month=month,
                run_root=run_root,
                layer="base",
                current=validation,
                features=base_features,
            )
        )
        rows.extend(
            _feature_drift_for_layer(
                eval_month=month,
                run_root=run_root,
                layer="meta",
                reference=meta_oof,
                current=validation,
                features=meta_features,
            )
        )
        rows.extend(
            _embedded_drift_for_layer(
                eval_month=month,
                run_root=run_root,
                layer="meta",
                reference=meta_oof,
                current=validation,
            )
        )
        rows.extend(
            _selected_feature_coverage_for_layer(
                eval_month=month,
                run_root=run_root,
                layer="meta",
                current=validation,
                features=meta_features,
            )
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["eval_month", "layer", "psi"], ascending=[True, True, False])
    return out


def build_meta_gate_attribution(runs: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_root in runs:
        _, meta = _month_from_run_id(run_root.name) or (None, None)
        if meta is None:
            continue
        policy = _policy_oos_frame(run_root)
        validation = policy[
            (policy["timestamp"] >= meta["validation_start"])
            & (policy["timestamp"] < meta["validation_end"])
        ].copy()
        for frac in TOP_FRACS:
            accepted_mask, degenerate = _top_mask(validation, "clf", frac)
            accepted = validation.loc[accepted_mask].copy()
            rejected = validation.loc[~accepted_mask].copy()
            acc_stats = _slice_stats(accepted, {})
            rej_stats = _slice_stats(rejected, {})
            rows.append(
                {
                    "eval_month": meta["eval_month"],
                    "run_id": run_root.name,
                    "gate": f"meta_top{int(frac * 100)}",
                    "score_degenerate": bool(degenerate),
                    "universe_rows": int(len(validation)),
                    "accepted_rows": int(len(accepted)),
                    "rejected_rows": int(len(rejected)),
                    "accepted_symbols": acc_stats["symbols"],
                    "rejected_symbols": rej_stats["symbols"],
                    "accepted_y_bin_rate": acc_stats["y_bin_rate"],
                    "rejected_y_bin_rate": rej_stats["y_bin_rate"],
                    "accepted_mean_return": acc_stats["mean_return"],
                    "rejected_mean_return": rej_stats["mean_return"],
                    "accepted_minus_rejected_mean_return": acc_stats["mean_return"] - rej_stats["mean_return"],
                    "accepted_hr_vn_2_1": acc_stats["hr_vn_2_1"],
                    "rejected_hr_vn_2_1": rej_stats["hr_vn_2_1"],
                    "accepted_minus_rejected_hr_vn_2_1": acc_stats["hr_vn_2_1"] - rej_stats["hr_vn_2_1"],
                    "accepted_hr_vn_3_2": acc_stats["hr_vn_3_2"],
                    "rejected_hr_vn_3_2": rej_stats["hr_vn_3_2"],
                    "accepted_base_score_mean": _safe_mean(accepted.get("oof_base_clf")),
                    "rejected_base_score_mean": _safe_mean(rejected.get("oof_base_clf")),
                    "accepted_meta_score_mean": _safe_mean(accepted.get("clf")),
                    "rejected_meta_score_mean": _safe_mean(rejected.get("clf")),
                }
            )
    return pd.DataFrame(rows)


def _extract_strategy_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    strategies = data.get("strategies", data)
    for key, value in strategies.items():
        if not isinstance(value, dict) or key.startswith("__"):
            continue
        return {
            "best_params": value.get("best_params", {}),
            "outer": value.get("outer_policy_validation_deployment_metrics", {}),
            "source_validation": value.get("source_validation", {}),
            "policy_outer_validation_rows": value.get("policy_outer_validation_rows"),
        }
    return {}


def build_policy_execution(report_root: Path, runs: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    failure = report_root / "failure_attribution_tests" / "execution_translation.csv"
    clean = report_root / "oos_month_week_diagnosis" / "clean_single_head_monthly.csv"
    vanilla = report_root / "vanilla_walkforward_oos_summary.csv"
    comparison = report_root / "policy_comparison" / "monthly_oos_policy_comparison.csv"
    if failure.exists():
        frame = pd.read_csv(failure)
        for _, row in frame.iterrows():
            rows.append(
                {
                    "eval_month": row.get("eval_month"),
                    "policy": "vanilla_execution_translation_replay",
                    "metric_source": str(failure),
                    "n_trades": row.get("selected_trades"),
                    "net_pnl": row.get("net_pnl"),
                    "mean_net_trade": row.get("sim_net_mean_return"),
                    "mean_gross_trade": row.get("sim_gross_mean_return"),
                    "selected_label_mean_return": row.get("selected_label_mean_return"),
                    "label_to_gross_drag": row.get("label_to_gross_mean_return_drag"),
                    "gross_to_net_drag": row.get("gross_to_net_mean_return_drag"),
                    "hit_rate": row.get("net_hit_rate"),
                    "gross_hit_rate": row.get("gross_hit_rate"),
                    "full_sl_exit_rate": row.get("full_sl_exit_rate"),
                    "trailing_exit_rate": row.get("trailing_exit_rate"),
                    "timeout_exit_rate": row.get("timeout_exit_rate"),
                }
            )
    if clean.exists():
        frame = pd.read_csv(clean)
        for _, row in frame.iterrows():
            rows.append(
                {
                    "eval_month": row.get("eval_month"),
                    "policy": "clean_single_head_vanilla_top15",
                    "metric_source": str(clean),
                    "n_trades": row.get("n_trades"),
                    "net_pnl": row.get("net_pnl"),
                    "mean_net_trade": row.get("mean_net"),
                    "hit_rate": row.get("hit_rate"),
                }
            )
    if vanilla.exists():
        frame = pd.read_csv(vanilla)
        frame = frame[frame["rank_slice"].eq("top_15")].copy()
        for _, row in frame.iterrows():
            month_meta = _month_from_run_id(str(row.get("run_id")))
            month = month_meta[1]["eval_month"] if month_meta else None
            rows.append(
                {
                    "eval_month": month,
                    "policy": "vanilla_walkforward_summary_top15",
                    "metric_source": str(vanilla),
                    "n_trades": row.get("n_trades"),
                    "candidate_rows": row.get("candidate_rows"),
                    "net_pnl": row.get("net_pnl"),
                    "mean_net_trade": row.get("mean_net_trade"),
                    "hit_rate": row.get("hit_rate"),
                    "max_drawdown": row.get("max_drawdown"),
                    "avg_holding_bars": row.get("avg_holding_bars"),
                    "full_sl_exit_rate": row.get("full_sl_exit_rate"),
                    "trailing_exit_rate": row.get("trailing_exit_rate"),
                    "timeout_exit_rate": row.get("timeout_exit_rate"),
                }
            )
    if comparison.exists():
        frame = pd.read_csv(comparison)
        frame = frame[
            frame["scope"].eq("single_head_monthly_walkforward")
            & frame["policy"].astype(str).str.contains("Optuna", na=False)
        ].copy()
        for _, row in frame.iterrows():
            rows.append(
                {
                    "eval_month": row.get("eval_month"),
                    "policy": "optuna_simple_policy_outer_validation",
                    "metric_source": str(comparison),
                    "n_trades": row.get("n_trades"),
                    "net_pnl": row.get("net_pnl"),
                    "mean_net_trade": row.get("mean_net_trade"),
                    "hit_rate": row.get("hit_rate"),
                    "max_drawdown": row.get("max_drawdown"),
                    "sortino": row.get("sortino"),
                }
            )
    for run_root in runs:
        _, meta = _month_from_run_id(run_root.name) or (None, None)
        if meta is None:
            continue
        metrics = _extract_strategy_metrics(run_root / "policy_optimisation_oos_metrics_perps.json")
        outer = metrics.get("outer", {})
        params = metrics.get("best_params", {})
        if outer:
            rows.append(
                {
                    "eval_month": meta["eval_month"],
                    "policy": "optuna_outer_validation_json",
                    "metric_source": str(run_root / "policy_optimisation_oos_metrics_perps.json"),
                    "n_trades": outer.get("n_trades"),
                    "net_pnl": outer.get("net_pnl"),
                    "mean_net_trade": outer.get("mean_net_trade"),
                    "mean_gross_trade": outer.get("mean_gross_trade"),
                    "gross_to_net_drag": (
                        outer.get("mean_gross_trade") - outer.get("mean_net_trade")
                        if outer.get("mean_gross_trade") is not None and outer.get("mean_net_trade") is not None
                        else None
                    ),
                    "mean_gross_trade_slippage_adjusted": outer.get("mean_gross_trade_slippage_adjusted"),
                    "gross_slippage_buffer": outer.get("gross_slippage_buffer"),
                    "hit_rate": outer.get("hit_rate"),
                    "max_drawdown": outer.get("max_drawdown"),
                    "sortino": outer.get("sortino"),
                    "avg_holding_bars": outer.get("avg_holding_bars"),
                    "p90_holding_bars": outer.get("p90_holding_bars"),
                    "sl_mult": params.get("sl_mult"),
                    "sl_abs_cap_pct": params.get("sl_abs_cap_pct"),
                    "trailing_activation_mult": params.get("trailing_activation_mult"),
                    "trailing_activation_cap_pct": params.get("trailing_activation_cap_pct"),
                    "giveback_beta": params.get("giveback_beta"),
                    "capital_protect_mfe_mult": params.get("capital_protect_mfe_mult"),
                    "target_holding_hours": params.get("target_holding_hours"),
                }
            )
    weekly = pd.read_csv(report_root / "oos_month_week_diagnosis" / "clean_single_head_vanilla_weekly.csv")
    return pd.DataFrame(rows), weekly


def _bucketed(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    out = pd.Series("missing", index=frame.index, dtype=object)
    valid = values.notna()
    if int(valid.sum()) < 10 or values[valid].nunique(dropna=True) < 2:
        out.loc[valid] = "all"
        return out
    try:
        out.loc[valid] = pd.qcut(values[valid], 3, labels=["low", "mid", "high"], duplicates="drop").astype(str)
    except ValueError:
        out.loc[valid] = "all"
    return out


def build_regime_universe(runs: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    symbol_rows: list[dict[str, Any]] = []
    for run_root in runs:
        _, meta = _month_from_run_id(run_root.name) or (None, None)
        if meta is None:
            continue
        month = meta["eval_month"]
        policy = _policy_oos_frame(run_root)
        validation = policy[
            (policy["timestamp"] >= meta["validation_start"])
            & (policy["timestamp"] < meta["validation_end"])
        ].copy()
        manifest_path = run_root / "policy_oos_predictions" / "manifest.json"
        outside_symbols = 0
        trained_symbols = np.nan
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            written = manifest.get("written", [])
            if written:
                filt = written[0].get("trained_universe_filter", {})
                outside_symbols = int(filt.get("dropped_symbols", 0) or 0)
                trained_symbols = filt.get("trained_universe_symbols")
        rows.append(
            _slice_stats(
                validation,
                {
                    "eval_month": month,
                    "run_id": run_root.name,
                    "dimension": "trained_universe",
                    "bucket": "trained_symbols_only" if outside_symbols == 0 else "mixed",
                    "outside_symbols": outside_symbols,
                    "trained_universe_symbols": trained_symbols,
                },
            )
        )
        for col in [c for c in REGIME_COLS if c in validation.columns]:
            work = validation.copy()
            work["_bucket"] = _bucketed(work, col)
            for bucket, group in work.groupby("_bucket", observed=True):
                rows.append(
                    _slice_stats(
                        group,
                        {
                            "eval_month": month,
                            "run_id": run_root.name,
                            "dimension": col,
                            "bucket": str(bucket),
                            "outside_symbols": outside_symbols,
                            "trained_universe_symbols": trained_symbols,
                        },
                    )
                )
        top_mask, _ = _top_mask(validation, "clf", 0.15)
        for symbol, group in validation.loc[top_mask].groupby("symbol", observed=True):
            if len(group) < 2:
                continue
            stats = _slice_stats(group, {})
            symbol_rows.append(
                {
                    "eval_month": month,
                    "symbol": symbol,
                    "slice": "meta_top15",
                    "rows": stats["rows"],
                    "mean_return": stats["mean_return"],
                    "sum_return": stats["sum_return"],
                    "y_bin_rate": stats["y_bin_rate"],
                    "hr_vn_2_1": stats["hr_vn_2_1"],
                    "hr_vn_3_2": stats["hr_vn_3_2"],
                }
            )
    regimes = pd.DataFrame(rows)
    symbols = pd.DataFrame(symbol_rows)
    if not symbols.empty:
        symbols = symbols.sort_values(["eval_month", "sum_return"], ascending=[True, True])
    return regimes, symbols


def _variant_series(frame: pd.DataFrame) -> dict[str, pd.Series]:
    ret = pd.to_numeric(frame.get("return", frame.get("y_ret")), errors="coerce")
    bars = pd.to_numeric(frame.get("bars_to_mfe"), errors="coerce")
    variants = {
        "current_y_bin": pd.to_numeric(frame.get("y_bin"), errors="coerce"),
        "return_gt_0": (ret > 0.0).astype(float),
        "return_gt_20bps": (ret > 0.002).astype(float),
        "return_gt_50bps": (ret > 0.005).astype(float),
        "mfe_ge_1barrier": pd.to_numeric(frame.get("mfe_ge_1"), errors="coerce"),
        "vn_tp2_sl1": pd.to_numeric(frame.get("hit_vn_2_1"), errors="coerce"),
        "vn_tp3_sl2": pd.to_numeric(frame.get("hit_vn_3_2"), errors="coerce"),
        "drawdown_ok_mae_lt1": (pd.to_numeric(frame.get("mae_norm"), errors="coerce") < 1.0).astype(float),
        "fast_mfe_ge1_le3bars": (
            (pd.to_numeric(frame.get("mfe_norm"), errors="coerce") >= 1.0) & (bars <= 3.0)
        ).astype(float),
    }
    if "u_policy_net" in frame.columns:
        variants["u_policy_net_gt0"] = (pd.to_numeric(frame["u_policy_net"], errors="coerce") > 0.0).astype(float)
    return variants


def _binary_lift_row(
    *,
    eval_month: str,
    run_id: str,
    layer: str,
    score_col: str,
    variant: str,
    target: pd.Series,
    frame: pd.DataFrame,
) -> dict[str, Any]:
    score = pd.to_numeric(frame.get(score_col), errors="coerce")
    valid = score.notna() & target.notna()
    work = frame.loc[valid].copy()
    target = target.loc[valid]
    base = _safe_mean(target)
    row: dict[str, Any] = {
        "eval_month": eval_month,
        "run_id": run_id,
        "layer": layer,
        "score_col": score_col,
        "variant": variant,
        "rows": int(len(work)),
        "base_rate": base,
        "ic": _safe_spearman(score.loc[valid], target),
        "score_unique": int(score.loc[valid].nunique(dropna=True)),
    }
    for frac in (0.30, 0.10):
        mask, degenerate = _top_mask(work, score_col, frac)
        top_target = target.loc[mask]
        top_rate = _safe_mean(top_target)
        tag = int(frac * 100)
        row[f"top{tag}_rate"] = top_rate
        row[f"top{tag}_lift"] = top_rate / base if base and math.isfinite(base) else float("nan")
        row[f"top{tag}_rows"] = int(mask.sum())
        row[f"top{tag}_score_degenerate"] = bool(degenerate)
    return row


def build_label_variants(runs: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_root in runs:
        _, meta = _month_from_run_id(run_root.name) or (None, None)
        if meta is None:
            continue
        policy = _policy_oos_frame(run_root)
        validation = policy[
            (policy["timestamp"] >= meta["validation_start"])
            & (policy["timestamp"] < meta["validation_end"])
        ].copy()
        for variant, target in _variant_series(validation).items():
            for layer, score_col in (("base", "oof_base_clf"), ("meta", "clf")):
                rows.append(
                    _binary_lift_row(
                        eval_month=meta["eval_month"],
                        run_id=run_root.name,
                        layer=layer,
                        score_col=score_col,
                        variant=variant,
                        target=target,
                        frame=validation,
                    )
                )
    return pd.DataFrame(rows)


def _write_markdown(
    output_dir: Path,
    *,
    label_economics: pd.DataFrame,
    feature_drift: pd.DataFrame,
    model_translation: pd.DataFrame,
    meta_gate: pd.DataFrame,
    policy_execution: pd.DataFrame,
    regime: pd.DataFrame,
    label_variants: pd.DataFrame,
) -> Path:
    md_path = output_dir / "walkforward_diagnostics_first5_7_9.md"

    label_view_cols = [
        "eval_month",
        "slice",
        "rows",
        "y_bin_rate",
        "mean_return",
        "hr_vn_2_1",
        "hr_vn_3_2",
        "mfe_norm_mean",
        "mae_norm_mean",
    ]
    label_view = label_economics[
        label_economics["period_type"].eq("month")
        & label_economics["slice"].isin(["all_base_gate_top40", "base_top10", "meta_top10", "meta_top15"])
    ][label_view_cols].copy()

    drift_view = feature_drift.head(20)[
        [
            "eval_month",
            "layer",
            "feature",
            "psi",
            "ks",
            "reference_ic_return",
            "current_ic_return",
            "reference_finite_rate",
            "current_finite_rate",
        ]
    ].copy() if not feature_drift.empty else pd.DataFrame()

    model_view_cols = [
        "eval_month",
        "layer",
        "rows",
        "lift_at_30",
        "lift_at_10",
        "ic_return",
        "ic_y_bin",
        "decile_spearman_return",
        "auc",
        "mean_return_at_10",
    ]
    model_view = model_translation[model_translation["sample"].eq("policy_oos_validation")][model_view_cols].copy()

    gate_view_cols = [
        "eval_month",
        "gate",
        "score_degenerate",
        "accepted_rows",
        "rejected_rows",
        "accepted_mean_return",
        "rejected_mean_return",
        "accepted_minus_rejected_mean_return",
        "accepted_hr_vn_2_1",
        "rejected_hr_vn_2_1",
    ]
    gate_view = meta_gate[gate_view_cols].copy()

    exec_view_cols = [
        "eval_month",
        "policy",
        "n_trades",
        "net_pnl",
        "mean_net_trade",
        "mean_gross_trade",
        "selected_label_mean_return",
        "label_to_gross_drag",
        "gross_to_net_drag",
        "full_sl_exit_rate",
        "trailing_exit_rate",
    ]
    exec_view = policy_execution[[c for c in exec_view_cols if c in policy_execution.columns]].copy()

    regime_view = regime[
        regime["dimension"].isin(
            ["trained_universe", "realized_volatility_24h", "median_spread_bps", "meta_lgbm_inference_drift_score"]
        )
    ][["eval_month", "dimension", "bucket", "rows", "mean_return", "y_bin_rate", "hr_vn_2_1", "hr_vn_3_2"]].copy()

    variant_view = label_variants[
        label_variants["variant"].isin(["current_y_bin", "return_gt_20bps", "vn_tp2_sl1", "fast_mfe_ge1_le3bars"])
    ][["eval_month", "layer", "variant", "base_rate", "top30_lift", "top10_lift", "ic"]].copy()

    for frame in (label_view, drift_view, model_view, gate_view, exec_view, regime_view, variant_view):
        for col in frame.columns:
            if pd.api.types.is_float_dtype(frame[col]):
                frame[col] = frame[col].map(lambda v: _fmt(v, 4))

    lines = [
        "# Walk-Forward Diagnostics 1-5, 7, 9",
        "",
        "Scope: Apr/May/Jun 2026 single-head monthly walk-forward, exact OOS validation windows.",
        "",
        "## 1. Label Economics",
        "",
        label_view.to_markdown(index=False),
        "",
        "## 2. Feature Drift / Coverage",
        "",
        drift_view.to_markdown(index=False) if not drift_view.empty else "No feature drift rows.",
        "",
        "## 3. Model OOF-To-OOS Translation",
        "",
        model_view.to_markdown(index=False),
        "",
        "## 4. Meta Gate Attribution",
        "",
        gate_view.to_markdown(index=False),
        "",
        "## 5. Policy Execution Decomposition",
        "",
        exec_view.to_markdown(index=False),
        "",
        "## 7. Regime / Universe Decomposition",
        "",
        regime_view.to_markdown(index=False),
        "",
        "## 9. Label Variant Replay",
        "",
        variant_view.to_markdown(index=False),
        "",
        "CSV files in this directory contain the full tables.",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


def build_report(experiment_id: str, data_root: Path, output_dir: Path) -> dict[str, str]:
    artifact_root = data_root / "artifacts"
    report_root = data_root / "reports" / experiment_id
    runs = sorted(
        (
            p
            for p in artifact_root.glob(f"{experiment_id}_train_*_score_*")
            if p.is_dir() and _month_from_run_id(p.name) is not None
        ),
        key=lambda p: (_month_from_run_id(p.name) or ("", {"validation_start": pd.Timestamp.max.tz_localize("UTC")}))[1][
            "validation_start"
        ],
    )
    if not runs:
        raise FileNotFoundError(f"No monthly walk-forward runs found for {experiment_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    label_economics = build_label_economics(runs)
    feature_drift = build_feature_drift(runs)
    model_paths = model_metrics.build_report(
        experiment_id=experiment_id,
        data_root=data_root,
        output_dir=output_dir / "model_translation",
    )
    model_translation = pd.read_csv(model_paths["metrics"])
    meta_gate = build_meta_gate_attribution(runs)
    policy_execution, policy_execution_weekly = build_policy_execution(report_root, runs)
    regime, symbol_breakdown = build_regime_universe(runs)
    label_variants = build_label_variants(runs)

    paths = {
        "label_economics": output_dir / "label_economics_month_week.csv",
        "feature_drift": output_dir / "feature_drift_coverage.csv",
        "model_translation": output_dir / "model_translation_metrics.csv",
        "meta_gate_attribution": output_dir / "meta_gate_attribution.csv",
        "policy_execution": output_dir / "policy_execution_decomposition.csv",
        "policy_execution_weekly": output_dir / "policy_execution_weekly.csv",
        "regime_universe": output_dir / "regime_universe_decomposition.csv",
        "symbol_breakdown": output_dir / "symbol_breakdown_meta_top15.csv",
        "label_variants": output_dir / "label_variant_replay.csv",
    }
    label_economics.to_csv(paths["label_economics"], index=False)
    feature_drift.to_csv(paths["feature_drift"], index=False)
    model_translation.to_csv(paths["model_translation"], index=False)
    meta_gate.to_csv(paths["meta_gate_attribution"], index=False)
    policy_execution.to_csv(paths["policy_execution"], index=False)
    policy_execution_weekly.to_csv(paths["policy_execution_weekly"], index=False)
    regime.to_csv(paths["regime_universe"], index=False)
    symbol_breakdown.to_csv(paths["symbol_breakdown"], index=False)
    label_variants.to_csv(paths["label_variants"], index=False)
    md_path = _write_markdown(
        output_dir,
        label_economics=label_economics,
        feature_drift=feature_drift,
        model_translation=model_translation,
        meta_gate=meta_gate,
        policy_execution=policy_execution,
        regime=regime,
        label_variants=label_variants,
    )

    manifest = {
        "experiment_id": experiment_id,
        "report_root": str(report_root),
        "output_dir": str(output_dir),
        "runs": [p.name for p in runs],
        "model_translation_subreport": model_paths,
        "outputs": {k: str(v) for k, v in paths.items()},
        "markdown": str(md_path),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return {**{k: str(v) for k, v in paths.items()}, "markdown": str(md_path), "manifest": str(manifest_path)}


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
        / "diagnostics_first5_7_9"
    )
    print(json.dumps(build_report(str(args.experiment_id), args.data_root, output_dir), indent=2))


if __name__ == "__main__":
    main()
