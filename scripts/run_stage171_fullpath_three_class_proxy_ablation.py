#!/usr/bin/env python3
"""No-training three-class full-path label proxy diagnostic.

This tests whether causal features can separate:

1. clean first-touch plus clean full-path continuation,
2. clean first-touch but dirty full-path reversal,
3. dirty first-touch.

It intentionally does not fit a model. For each holdout month, it chooses a
small signed feature ensemble using only prior rows, scores holdout rows with
train-only medians/IQRs, and reports class/economic metrics for top buckets.
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

from scripts.run_first_touch_label_training_smoke import (  # noqa: E402
    _first_touch_eval_metrics,
    _table,
)
from scripts.run_label_quality_proxy_diagnostics import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    DEFAULT_FEATURE_LIST_CSV,
    _json_safe,
    _load_feature_store_columns,
    _load_labels,
    _path_metrics,
    _read_feature_list,
    _safe_mean,
    _safe_quantile,
    _spearman,
)


DEFAULT_LABELS_PATH = Path("data_perp/artifacts/20260703_190000_clean_first_touch_tail_veto_stage167_labels/labels")
DEFAULT_OUTPUT_DIR = Path("data_perp/reports/stage171_fullpath_three_class_proxy_v1")
DEFAULT_EXTRA_FEATURE_CSV = Path(
    "data_perp/reports/stage167_full_path_tail_feature_gap_v1/"
    "stage167_full_path_tail_feature_contrast.csv"
)
DEFAULT_MONTHS = ("2026-04", "2026-05", "2026-06")
DEFAULT_TOP_FRACS = (0.005, 0.01, 0.03, 0.05, 0.10, 0.35, 0.50, 0.65, 0.80)


def _safe_numeric(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _parse_csv(value: str | list[str] | tuple[str, ...], default: tuple[str, ...] = ()) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(part).strip() for part in value if str(part).strip()]
    text = str(value).strip()
    if not text:
        return list(default)
    return [part.strip() for part in text.split(",") if part.strip()]


def _parse_float_csv(value: str | list[float] | tuple[float, ...]) -> list[float]:
    if isinstance(value, (list, tuple)):
        return [float(part) for part in value]
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def _auc_binary(score: Any, target: Any) -> float:
    s = _safe_numeric(score)
    y = _safe_numeric(target)
    mask = s.notna() & y.notna()
    if int(mask.sum()) < 10:
        return float("nan")
    yb = y[mask] > 0.5
    n_pos = int(yb.sum())
    n_neg = int((~yb).sum())
    if n_pos < 3 or n_neg < 3:
        return float("nan")
    ranks = s[mask].rank(method="average")
    rank_sum_pos = float(ranks[yb].sum())
    return float((rank_sum_pos - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg))


def _decile_monotonicity(score: Any, target: Any) -> float:
    s = _safe_numeric(score)
    y = _safe_numeric(target)
    mask = s.notna() & y.notna()
    if int(mask.sum()) < 50:
        return float("nan")
    try:
        decile = pd.qcut(s[mask].rank(method="first"), 10, labels=False, duplicates="drop")
    except ValueError:
        return float("nan")
    grouped = y[mask].groupby(decile).mean()
    if len(grouped) < 4:
        return float("nan")
    return _spearman(pd.Series(grouped.index, dtype=float), grouped.reset_index(drop=True))


def _read_extra_features(path: Path | None, *, max_features: int) -> list[str]:
    if path is None or not str(path) or str(path) == "/dev/null" or not path.exists():
        return []
    frame = pd.read_csv(path)
    if "feature" not in frame.columns:
        return []
    if "best_auc" in frame.columns:
        frame = frame.sort_values("best_auc", ascending=False)
    return [str(v) for v in frame["feature"].dropna().drop_duplicates().head(max_features).tolist()]


def _load_feature_names(
    *,
    feature_list_csv: Path,
    extra_feature_csv: Path | None,
    max_features: int,
    max_extra_features: int,
) -> list[str]:
    base = _read_feature_list(feature_list_csv, max_features=max_features)
    extra = _read_extra_features(extra_feature_csv, max_features=max_extra_features)
    return list(dict.fromkeys([*base, *extra]))


def _attach_features(
    frame: pd.DataFrame,
    *,
    feature_dir: Path,
    features: list[str],
) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    matrix, manifest = _load_feature_store_columns(
        frame,
        feature_dir=feature_dir,
        selected_features=features,
    )
    out = frame.reset_index(drop=True).copy()
    if not matrix.empty:
        new_cols = [col for col in matrix.columns if col not in out.columns]
        if new_cols:
            out = pd.concat([out, matrix.loc[:, new_cols].reset_index(drop=True)], axis=1)
    usable = [feature for feature in features if feature in out.columns]
    return out, manifest, usable


def _label_classes(
    metrics: pd.DataFrame,
    *,
    first_touch_clean_r: float,
    full_path_clean_r: float,
    full_path_dirty_r: float,
) -> pd.DataFrame:
    first_mae = _safe_numeric(metrics["first_touch_mae_to_sl"]).fillna(10.0)
    full_mae = _safe_numeric(metrics["first_touch_full_path_mae_to_sl"]).fillna(10.0)
    clean_exec = _safe_numeric(metrics["clean_first_touch_exec"]).fillna(0.0) >= 0.5
    timeout = _safe_numeric(metrics["first_touch_timeout"]).fillna(1.0) >= 0.5
    net = _safe_numeric(metrics["first_touch_net"]).fillna(-0.05)
    first_touch_clean = clean_exec & (~timeout) & (net > 0.0) & (first_mae <= float(first_touch_clean_r))
    clean_continue = first_touch_clean & (full_mae <= float(full_path_clean_r))
    clean_reversal = first_touch_clean & (full_mae >= float(full_path_dirty_r))
    dirty_first_touch = ~first_touch_clean
    class_id = pd.Series(0, index=metrics.index, dtype=np.int8)
    class_id.loc[clean_reversal] = 1
    class_id.loc[clean_continue] = 2
    return pd.DataFrame(
        {
            "first_touch_clean": first_touch_clean.astype(float),
            "clean_continue": clean_continue.astype(float),
            "clean_reversal": clean_reversal.astype(float),
            "dirty_first_touch": dirty_first_touch.astype(float),
            "class_id": class_id,
            "three_class_utility": (
                clean_continue.astype(float)
                - clean_reversal.astype(float)
                - 0.50 * dirty_first_touch.astype(float)
            ),
        },
        index=metrics.index,
    )


def _selected_top_mask(score: pd.Series, frac: float) -> pd.Series:
    out = pd.Series(False, index=score.index)
    values = _safe_numeric(score)
    valid = values.notna().to_numpy()
    if not bool(valid.any()):
        return out
    valid_idx = np.flatnonzero(valid)
    k = max(1, int(math.ceil(float(frac) * len(valid_idx))))
    k = min(k, len(valid_idx))
    order = np.argsort(-values.iloc[valid_idx].to_numpy(dtype=np.float64), kind="mergesort")
    out.iloc[valid_idx[order[:k]]] = True
    return out


def _fit_proxy(
    train: pd.DataFrame,
    *,
    features: list[str],
    target_col: str,
    top_k: int,
    min_rows: int,
    min_abs_ic: float,
) -> tuple[list[dict[str, Any]], pd.Series]:
    y = _safe_numeric(train[target_col])
    rows: list[dict[str, Any]] = []
    for feature in features:
        if feature not in train.columns:
            continue
        x = _safe_numeric(train[feature]).replace([np.inf, -np.inf], np.nan)
        mask = x.notna() & y.notna()
        if int(mask.sum()) < int(min_rows) or int(y[mask].nunique(dropna=True)) < 2 or int(x[mask].nunique(dropna=True)) < 4:
            continue
        ic = _spearman(x[mask], y[mask])
        auc = _auc_binary(x[mask], y[mask]) if set(y[mask].dropna().unique()).issubset({0.0, 1.0}) else float("nan")
        if not math.isfinite(ic):
            continue
        median = float(x[mask].median())
        q25 = float(x[mask].quantile(0.25))
        q75 = float(x[mask].quantile(0.75))
        scale = float(q75 - q25)
        if not math.isfinite(scale) or scale <= 1e-8:
            scale = float(x[mask].std(ddof=0))
        if not math.isfinite(scale) or scale <= 1e-8:
            continue
        rows.append(
            {
                "feature": feature,
                "train_ic": float(ic),
                "train_abs_ic": abs(float(ic)),
                "train_auc": auc,
                "sign": 1.0 if ic >= 0.0 else -1.0,
                "median": median,
                "scale": scale,
                "finite_rows": int(mask.sum()),
            }
        )
    ranked = sorted(rows, key=lambda row: (row["train_abs_ic"], abs(float(row.get("train_auc", 0.5) or 0.5) - 0.5)), reverse=True)
    if min_abs_ic > 0.0:
        ranked = [row for row in ranked if float(row["train_abs_ic"]) >= float(min_abs_ic)]
    params = ranked[: int(top_k)]
    return params, _score_proxy(train, params)


def _score_proxy(frame: pd.DataFrame, params: list[dict[str, Any]]) -> pd.Series:
    if not params:
        return pd.Series(0.0, index=frame.index)
    parts = []
    for param in params:
        values = _safe_numeric(frame[param["feature"]]).replace([np.inf, -np.inf], np.nan)
        z = ((values - float(param["median"])) / float(param["scale"])).clip(-5.0, 5.0)
        parts.append(float(param["sign"]) * z)
    return pd.concat(parts, axis=1).mean(axis=1).fillna(0.0)


def _summary_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    rows = int(len(frame))
    full_mae = _safe_numeric(frame.get("first_touch_full_path_mae_to_sl"))
    first_mae = _safe_numeric(frame.get("first_touch_mae_to_sl"))
    return {
        "rows": rows,
        "first_touch_clean_rate": _safe_mean(frame.get("first_touch_clean")),
        "clean_continue_rate": _safe_mean(frame.get("clean_continue")),
        "clean_reversal_rate": _safe_mean(frame.get("clean_reversal")),
        "dirty_first_touch_rate": _safe_mean(frame.get("dirty_first_touch")),
        "mean_first_touch_net": _safe_mean(frame.get("first_touch_net")),
        "sum_first_touch_net": float(_safe_numeric(frame.get("first_touch_net")).sum()) if rows else 0.0,
        "bad_first_touch_mae_to_sl_rate": _safe_mean(first_mae >= 1.0),
        "first_touch_timeout_rate": _safe_mean(_safe_numeric(frame.get("first_touch_timeout")) >= 0.5),
        "bad_full_path_mae_3r_rate": _safe_mean(full_mae >= 3.0),
        "p90_full_path_mae_to_sl": _safe_quantile(full_mae, 0.90),
        "top_symbol_share": float(frame["__symbol__"].astype(str).value_counts(normalize=True).iloc[0])
        if rows and "__symbol__" in frame.columns
        else float("nan"),
    }


def _proxy_selection_rows(
    *,
    month: str,
    scope: str,
    valid: pd.DataFrame,
    score: pd.Series,
    target_col: str,
    top_fracs: list[float],
    train_params: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    baseline = _summary_metrics(valid)
    score_ic = _spearman(score, valid[target_col]) if target_col in valid.columns else float("nan")
    score_auc = _auc_binary(score, valid[target_col]) if target_col in valid.columns else float("nan")
    monotonicity = _decile_monotonicity(score, valid[target_col]) if target_col in valid.columns else float("nan")
    for frac in top_fracs:
        mask = _selected_top_mask(score.reset_index(drop=True), float(frac))
        selected = valid.reset_index(drop=True).loc[mask].copy()
        metrics = _summary_metrics(selected)
        row: dict[str, Any] = {
            "period": str(month),
            "scope": str(scope),
            "target_col": str(target_col),
            "top_frac": float(frac),
            "proxy_feature_count": int(len(train_params)),
            "proxy_features": ",".join([str(param["feature"]) for param in train_params]),
            "valid_score_ic": score_ic,
            "valid_score_auc": score_auc,
            "valid_decile_monotonicity": monotonicity,
        }
        for key, value in baseline.items():
            row[f"baseline_{key}"] = value
        for key, value in metrics.items():
            row[key] = value
        row["delta_sum_first_touch_net"] = float(row["sum_first_touch_net"]) - float(row["baseline_sum_first_touch_net"])
        row["lift_clean_continue_rate"] = (
            float(row["clean_continue_rate"]) / float(row["baseline_clean_continue_rate"])
            if float(row["baseline_clean_continue_rate"] or 0.0) > 0.0
            else float("nan")
        )
        row["delta_clean_reversal_rate"] = float(row["clean_reversal_rate"]) - float(row["baseline_clean_reversal_rate"])
        row["delta_bad_full_path_mae_3r_rate"] = (
            float(row["bad_full_path_mae_3r_rate"]) - float(row["baseline_bad_full_path_mae_3r_rate"])
        )
        rows.append(row)
    return rows


def _feature_rows(*, month: str, scope: str, params: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for rank, param in enumerate(params, start=1):
        row = {"period": str(month), "scope": str(scope), "feature_rank": int(rank)}
        row.update(param)
        out.append(row)
    return out


def _month_baseline_rows(month: str, frame: pd.DataFrame) -> dict[str, Any]:
    metrics = _summary_metrics(frame)
    row = {"period": str(month)}
    row.update(metrics)
    return row


def _write_markdown(
    *,
    output_dir: Path,
    baseline: pd.DataFrame,
    selection: pd.DataFrame,
    features: pd.DataFrame,
    manifest: dict[str, Any],
) -> Path:
    path = output_dir / "stage171_fullpath_three_class_proxy.md"
    baseline_cols = [
        "period",
        "rows",
        "first_touch_clean_rate",
        "clean_continue_rate",
        "clean_reversal_rate",
        "dirty_first_touch_rate",
        "mean_first_touch_net",
        "bad_full_path_mae_3r_rate",
        "p90_full_path_mae_to_sl",
    ]
    selection_cols = [
        "period",
        "scope",
        "target_col",
        "top_frac",
        "rows",
        "valid_score_ic",
        "valid_score_auc",
        "valid_decile_monotonicity",
        "clean_continue_rate",
        "clean_reversal_rate",
        "lift_clean_continue_rate",
        "mean_first_touch_net",
        "bad_full_path_mae_3r_rate",
        "p90_full_path_mae_to_sl",
        "proxy_features",
    ]
    feature_cols = [
        "period",
        "scope",
        "feature_rank",
        "feature",
        "train_ic",
        "train_auc",
        "finite_rows",
    ]
    focus = selection[
        selection["top_frac"].isin([0.005, 0.01, 0.05, 0.50, 0.80])
    ].copy() if not selection.empty else selection
    lines = [
        "# Stage171 Full-Path Three-Class Proxy Diagnostic",
        "",
        "Scope: no-training label-quality diagnostic. Feature proxies are selected on prior rows only and scored with train-only medians/IQRs.",
        "",
        f"Labels: `{manifest['labels_path']}`",
        f"Feature dir: `{manifest['feature_dir']}`",
        f"Feature count: `{manifest['feature_count']}`",
        f"Months: `{', '.join(manifest['months'])}`",
        f"Classes: clean continuation <= `{manifest['full_path_clean_r']}R`, dirty reversal >= `{manifest['full_path_dirty_r']}R`, dirty first-touch otherwise.",
        "",
        "## Monthly Class Baseline",
        "",
        _table(baseline, baseline_cols, limit=80),
        "",
        "## Proxy Selection Metrics",
        "",
        _table(focus, selection_cols, limit=120),
        "",
        "## Top Proxy Features",
        "",
        _table(features, feature_cols, limit=120),
        "",
        "## Outputs",
        "",
        f"- Baseline: `{manifest['outputs']['baseline']}`",
        f"- Selection metrics: `{manifest['outputs']['selection']}`",
        f"- Proxy features: `{manifest['outputs']['features']}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_ablation(
    *,
    labels_path: Path,
    output_dir: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    extra_feature_csv: Path | None,
    max_features: int,
    max_extra_features: int,
    months: list[str],
    top_fracs: list[float],
    proxy_top_k: int,
    min_proxy_rows: int,
    min_abs_ic: float,
    first_touch_clean_r: float,
    full_path_clean_r: float,
    full_path_dirty_r: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _load_labels(labels_path)
    metrics = _first_touch_eval_metrics(frame, _path_metrics(frame))
    classes = _label_classes(
        metrics,
        first_touch_clean_r=first_touch_clean_r,
        full_path_clean_r=full_path_clean_r,
        full_path_dirty_r=full_path_dirty_r,
    )
    model_frame = pd.concat(
        [
            frame.reset_index(drop=True),
            metrics[
                [
                    "first_touch_net",
                    "first_touch_timeout",
                    "first_touch_mae_to_sl",
                    "first_touch_full_path_mae_to_sl",
                    "clean_first_touch_exec",
                ]
            ].reset_index(drop=True),
            classes.reset_index(drop=True),
        ],
        axis=1,
    )
    feature_names = _load_feature_names(
        feature_list_csv=feature_list_csv,
        extra_feature_csv=extra_feature_csv,
        max_features=max_features,
        max_extra_features=max_extra_features,
    )
    model_frame, feature_manifest, features = _attach_features(
        model_frame,
        feature_dir=feature_dir,
        features=feature_names,
    )
    month_ser = model_frame["__ts__"].dt.to_period("M").astype(str)
    baseline_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []

    for month in months:
        train_mask = month_ser < str(month)
        valid_mask = month_ser == str(month)
        train = model_frame.loc[train_mask].copy()
        valid = model_frame.loc[valid_mask].copy()
        if train.empty or valid.empty:
            continue
        baseline_rows.append(_month_baseline_rows(str(month), valid))

        params_global, _train_score = _fit_proxy(
            train,
            features=features,
            target_col="clean_continue",
            top_k=proxy_top_k,
            min_rows=min_proxy_rows,
            min_abs_ic=min_abs_ic,
        )
        feature_rows.extend(_feature_rows(month=str(month), scope="global_clean_continue", params=params_global))
        valid_score = _score_proxy(valid, params_global)
        selection_rows.extend(
            _proxy_selection_rows(
                month=str(month),
                scope="global_clean_continue",
                valid=valid,
                score=valid_score,
                target_col="clean_continue",
                top_fracs=top_fracs,
                train_params=params_global,
            )
        )

        params_three_class, _ = _fit_proxy(
            train,
            features=features,
            target_col="three_class_utility",
            top_k=proxy_top_k,
            min_rows=min_proxy_rows,
            min_abs_ic=min_abs_ic,
        )
        feature_rows.extend(_feature_rows(month=str(month), scope="global_three_class_utility", params=params_three_class))
        three_class_score = _score_proxy(valid, params_three_class)
        selection_rows.extend(
            _proxy_selection_rows(
                month=str(month),
                scope="global_three_class_utility",
                valid=valid,
                score=three_class_score,
                target_col="three_class_utility",
                top_fracs=top_fracs,
                train_params=params_three_class,
            )
        )

        train_oracle = train[_safe_numeric(train["first_touch_clean"]) >= 0.5].copy()
        valid_oracle = valid[_safe_numeric(valid["first_touch_clean"]) >= 0.5].copy()
        if len(train_oracle) >= int(min_proxy_rows) and len(valid_oracle) >= 20:
            params_oracle, _ = _fit_proxy(
                train_oracle,
                features=features,
                target_col="clean_continue",
                top_k=proxy_top_k,
                min_rows=min_proxy_rows,
                min_abs_ic=min_abs_ic,
            )
            feature_rows.extend(
                _feature_rows(month=str(month), scope="oracle_firsttouch_continue_vs_reversal", params=params_oracle)
            )
            oracle_score = _score_proxy(valid_oracle, params_oracle)
            selection_rows.extend(
                _proxy_selection_rows(
                    month=str(month),
                    scope="oracle_firsttouch_continue_vs_reversal",
                    valid=valid_oracle,
                    score=oracle_score,
                    target_col="clean_continue",
                    top_fracs=[frac for frac in top_fracs if frac >= 0.10],
                    train_params=params_oracle,
                )
            )

    baseline = pd.DataFrame(baseline_rows)
    selection = pd.DataFrame(selection_rows)
    features_frame = pd.DataFrame(feature_rows)
    paths = {
        "baseline": output_dir / "stage171_three_class_monthly_baseline.csv",
        "selection": output_dir / "stage171_three_class_proxy_selection_metrics.csv",
        "features": output_dir / "stage171_three_class_proxy_features.csv",
        "manifest": output_dir / "manifest.json",
    }
    baseline.to_csv(paths["baseline"], index=False)
    selection.to_csv(paths["selection"], index=False)
    features_frame.to_csv(paths["features"], index=False)
    manifest = {
        "scope": "stage171_fullpath_three_class_proxy",
        "labels_path": str(labels_path),
        "output_dir": str(output_dir),
        "feature_dir": str(feature_dir),
        "feature_list_csv": str(feature_list_csv),
        "extra_feature_csv": str(extra_feature_csv) if extra_feature_csv else None,
        "feature_store": feature_manifest,
        "feature_count": int(len(features)),
        "features": features,
        "months": list(months),
        "top_fracs": list(top_fracs),
        "proxy_top_k": int(proxy_top_k),
        "min_proxy_rows": int(min_proxy_rows),
        "min_abs_ic": float(min_abs_ic),
        "first_touch_clean_r": float(first_touch_clean_r),
        "full_path_clean_r": float(full_path_clean_r),
        "full_path_dirty_r": float(full_path_dirty_r),
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    markdown = _write_markdown(output_dir=output_dir, baseline=baseline, selection=selection, features=features_frame, manifest=manifest)
    manifest["outputs"]["markdown"] = str(markdown)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--extra-feature-csv", type=Path, default=DEFAULT_EXTRA_FEATURE_CSV)
    parser.add_argument("--max-features", type=int, default=160)
    parser.add_argument("--max-extra-features", type=int, default=80)
    parser.add_argument("--months", default=",".join(DEFAULT_MONTHS))
    parser.add_argument("--top-fracs", default=",".join(str(v) for v in DEFAULT_TOP_FRACS))
    parser.add_argument("--proxy-top-k", type=int, default=8)
    parser.add_argument("--min-proxy-rows", type=int, default=500)
    parser.add_argument("--min-abs-ic", type=float, default=0.0)
    parser.add_argument("--first-touch-clean-r", type=float, default=1.0)
    parser.add_argument("--full-path-clean-r", type=float, default=3.0)
    parser.add_argument("--full-path-dirty-r", type=float, default=3.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_ablation(
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        extra_feature_csv=args.extra_feature_csv,
        max_features=int(args.max_features),
        max_extra_features=int(args.max_extra_features),
        months=_parse_csv(str(args.months), default=DEFAULT_MONTHS),
        top_fracs=_parse_float_csv(str(args.top_fracs)),
        proxy_top_k=int(args.proxy_top_k),
        min_proxy_rows=int(args.min_proxy_rows),
        min_abs_ic=float(args.min_abs_ic),
        first_touch_clean_r=float(args.first_touch_clean_r),
        full_path_clean_r=float(args.full_path_clean_r),
        full_path_dirty_r=float(args.full_path_dirty_r),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
