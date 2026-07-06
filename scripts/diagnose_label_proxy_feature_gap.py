#!/usr/bin/env python3
"""Diagnose why profitable oracle label rows are missed by proxy scores.

This is a no-training diagnostic. It compares an OOT month's oracle winners
against rows selected by a causal prior-month feature proxy, then reports the
feature differences and IC stability that explain the miss.
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

from scripts.run_label_economic_proxy_ablation import LABEL_ARMS, _label_targets
from scripts.run_label_quality_proxy_diagnostics import (
    DEFAULT_FEATURE_DIR,
    DEFAULT_FEATURE_LIST_CSV,
    DEFAULT_LABELS_DIR,
    _feature_columns,
    _json_safe,
    _load_feature_store_columns,
    _load_labels,
    _path_metrics,
    _rank_top_indices,
    _read_feature_list,
    _safe_mean,
    _safe_quantile,
    _selection_metrics,
    _spearman,
)
from scripts.run_label_weighted_proxy_ablation import (
    PROXY_METHODS,
    WEIGHT_ARMS,
    _effective_sample_size,
    _weight_series,
    _weighted_corr,
    _weighted_proxy_score,
)


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/label_proxy_feature_gap_june_v1")


def _safe_numeric(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _top_mask(score: pd.Series, frac: float) -> pd.Series:
    mask = pd.Series(False, index=score.index)
    idx = _rank_top_indices(score, frac)
    if len(idx):
        mask.iloc[idx] = True
    return mask


def _group_metrics(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    score: pd.Series,
    target: pd.DataFrame,
    mask: pd.Series,
    group: str,
    top_frac: float,
) -> dict[str, Any]:
    idx = np.flatnonzero(mask.to_numpy(dtype=bool))
    if len(idx):
        row = _selection_metrics(
            frame=frame.iloc[idx].reset_index(drop=True),
            metrics=metrics.iloc[idx].reset_index(drop=True),
            target=target.iloc[idx].reset_index(drop=True),
            score=score.iloc[idx].reset_index(drop=True),
            arm=group,
            selector="explicit_group",
            period="group",
            top_frac=1.0,
        )
    else:
        row = _selection_metrics(
            frame=frame.iloc[:0].reset_index(drop=True),
            metrics=metrics.iloc[:0].reset_index(drop=True),
            target=target.iloc[:0].reset_index(drop=True),
            score=score.iloc[:0].reset_index(drop=True),
            arm=group,
            selector="explicit_group",
            period="group",
            top_frac=1.0,
        )
    row["group"] = group
    row["group_frac"] = float(len(idx) / len(frame)) if len(frame) else float("nan")
    row["comparison_top_frac"] = float(top_frac)
    return row


def _rank_series(values: pd.Series) -> pd.Series:
    return _safe_numeric(values).rank(method="average", pct=True)


def _feature_contrasts(
    *,
    frame: pd.DataFrame,
    features: list[str],
    missed_mask: pd.Series,
    false_positive_mask: pd.Series,
    oracle_mask: pd.Series,
    proxy_mask: pd.Series,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in features:
        values = _safe_numeric(frame[feature])
        missed = values[missed_mask]
        false_pos = values[false_positive_mask]
        oracle = values[oracle_mask]
        proxy = values[proxy_mask]
        ranks = _rank_series(values)
        missed_ranks = ranks[missed_mask]
        false_pos_ranks = ranks[false_positive_mask]
        finite_missed = missed.dropna()
        finite_false = false_pos.dropna()
        if len(finite_missed) < 5 or len(finite_false) < 5:
            continue
        iqr = float(values.quantile(0.75) - values.quantile(0.25))
        median_gap = float(finite_missed.median() - finite_false.median())
        rank_gap = float(missed_ranks.mean() - false_pos_ranks.mean())
        rows.append(
            {
                "feature": feature,
                "missed_median": float(finite_missed.median()),
                "false_positive_median": float(finite_false.median()),
                "missed_minus_false_positive_median": median_gap,
                "robust_effect_iqr": median_gap / iqr if iqr > 0.0 else float("nan"),
                "missed_rank_mean": _safe_mean(missed_ranks),
                "false_positive_rank_mean": _safe_mean(false_pos_ranks),
                "missed_minus_false_positive_rank": rank_gap,
                "oracle_rank_mean": _safe_mean(ranks[oracle_mask]),
                "proxy_rank_mean": _safe_mean(ranks[proxy_mask]),
                "missed_finite_frac": float(missed.notna().mean()) if len(missed) else float("nan"),
                "false_positive_finite_frac": float(false_pos.notna().mean()) if len(false_pos) else float("nan"),
            }
        )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["abs_rank_gap"] = out["missed_minus_false_positive_rank"].abs()
    out["abs_robust_effect_iqr"] = out["robust_effect_iqr"].abs()
    return out.sort_values(["abs_rank_gap", "abs_robust_effect_iqr"], ascending=[False, False])


def _feature_ic_stability(
    *,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: list[str],
    target_train: pd.Series,
    target_valid: pd.Series,
    utility_train: pd.Series,
    utility_valid: pd.Series,
    weights: pd.Series,
    proxy_features: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    proxy_set = set(proxy_features)
    for feature in features:
        train_label_ic = _weighted_corr(train[feature], target_train, weights)
        valid_label_ic = _spearman(valid[feature], target_valid)
        train_u_ic = _spearman(train[feature], utility_train)
        valid_u_ic = _spearman(valid[feature], utility_valid)
        rows.append(
            {
                "feature": feature,
                "is_proxy_feature": feature in proxy_set,
                "train_weighted_label_ic": train_label_ic,
                "valid_label_ic": valid_label_ic,
                "train_utility_ic": train_u_ic,
                "valid_utility_ic": valid_u_ic,
                "label_ic_delta_valid_minus_train": (
                    valid_label_ic - train_label_ic
                    if math.isfinite(valid_label_ic) and math.isfinite(train_label_ic)
                    else float("nan")
                ),
                "utility_ic_delta_valid_minus_train": (
                    valid_u_ic - train_u_ic
                    if math.isfinite(valid_u_ic) and math.isfinite(train_u_ic)
                    else float("nan")
                ),
                "label_ic_sign_flip": (
                    bool(np.sign(train_label_ic) != np.sign(valid_label_ic))
                    if math.isfinite(train_label_ic) and math.isfinite(valid_label_ic)
                    else False
                ),
                "utility_ic_sign_flip": (
                    bool(np.sign(train_u_ic) != np.sign(valid_u_ic))
                    if math.isfinite(train_u_ic) and math.isfinite(valid_u_ic)
                    else False
                ),
            }
        )
    out = pd.DataFrame(rows)
    out["abs_valid_utility_ic"] = out["valid_utility_ic"].abs()
    out["abs_train_weighted_label_ic"] = out["train_weighted_label_ic"].abs()
    return out.sort_values(["is_proxy_feature", "abs_valid_utility_ic"], ascending=[False, False])


def _row_extract(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    score: pd.Series,
    target: pd.DataFrame,
    mask: pd.Series,
    features: list[str],
    limit: int,
) -> pd.DataFrame:
    cols = ["__ts__", "__symbol__"]
    feature_cols = features[: min(20, len(features))]
    out = frame.loc[mask, cols + feature_cols].copy()
    out["score"] = score[mask].to_numpy(dtype=np.float64, copy=False)
    out["target_soft"] = target.loc[mask, "target_soft"].to_numpy(dtype=np.float64, copy=False)
    for col in [
        "u_policy_net",
        "ret_net",
        "mfe_norm",
        "mae_norm",
        "barrier",
        "bars_to_mfe",
        "is_timeout",
    ]:
        values = metrics.loc[mask, col]
        if values.dtype == bool:
            out[col] = values.astype(int).to_numpy()
        else:
            out[col] = values.to_numpy()
    return out.sort_values("u_policy_net", ascending=False).head(limit)


def _write_markdown(
    *,
    output_dir: Path,
    group_summary: pd.DataFrame,
    feature_contrast: pd.DataFrame,
    ic_stability: pd.DataFrame,
    manifest: dict[str, Any],
) -> Path:
    path = output_dir / "label_proxy_feature_gap.md"

    def table(frame: pd.DataFrame, cols: list[str], limit: int | None = None) -> str:
        if frame.empty:
            return "No rows."
        view = frame[[c for c in cols if c in frame.columns]].copy()
        if limit is not None:
            view = view.head(limit)
        for col in view.columns:
            if pd.api.types.is_float_dtype(view[col]):
                view[col] = view[col].map(lambda v: f"{float(v):.4f}" if pd.notna(v) else "")
        return view.to_markdown(index=False)

    unstable = ic_stability[
        ic_stability["is_proxy_feature"]
        | ic_stability["utility_ic_sign_flip"]
        | ic_stability["label_ic_sign_flip"]
    ].copy()
    lines = [
        "# Label Proxy Feature Gap",
        "",
        "Scope: no model training. Compares OOT oracle winners with prior-month feature-proxy selections.",
        "",
        "## Group Summary",
        "",
        table(
            group_summary,
            [
                "group",
                "selected_rows",
                "group_frac",
                "mean_u",
                "hit_u",
                "q10_u",
                "bad_mae_1r_rate",
                "wide_barrier_25bps_rate",
                "top_symbol_share",
            ],
        ),
        "",
        "## Missed Winners Vs False Positives",
        "",
        table(
            feature_contrast,
            [
                "feature",
                "missed_rank_mean",
                "false_positive_rank_mean",
                "missed_minus_false_positive_rank",
                "robust_effect_iqr",
                "missed_median",
                "false_positive_median",
            ],
            limit=30,
        ),
        "",
        "## IC Stability",
        "",
        table(
            unstable.sort_values(
                ["is_proxy_feature", "utility_ic_sign_flip", "abs_valid_utility_ic"],
                ascending=[False, False, False],
            ),
            [
                "feature",
                "is_proxy_feature",
                "train_weighted_label_ic",
                "valid_label_ic",
                "train_utility_ic",
                "valid_utility_ic",
                "label_ic_sign_flip",
                "utility_ic_sign_flip",
            ],
            limit=40,
        ),
        "",
        "## Outputs",
        "",
        f"- Group summary: `{manifest['outputs']['group_summary']}`",
        f"- Feature contrast: `{manifest['outputs']['feature_contrast']}`",
        f"- IC stability: `{manifest['outputs']['ic_stability']}`",
        f"- Missed winners: `{manifest['outputs']['missed_winners']}`",
        f"- False positives: `{manifest['outputs']['false_positives']}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_diagnostic(
    *,
    labels_path: Path,
    output_dir: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    max_feature_store_features: int | None,
    month: str,
    label_arm: str,
    weight_arm: str,
    top_frac: float,
    row_limit: int,
    oracle_basis: str = "utility",
    proxy_method: str = "weighted_ic",
    tail_recovery_frac: float = 0.01,
) -> dict[str, Any]:
    if label_arm not in LABEL_ARMS:
        raise ValueError(f"label_arm must be one of {LABEL_ARMS}")
    if weight_arm not in WEIGHT_ARMS:
        raise ValueError(f"weight_arm must be one of {WEIGHT_ARMS}")
    if proxy_method not in PROXY_METHODS:
        raise ValueError(f"proxy_method must be one of {PROXY_METHODS}")
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = _load_labels(labels_path)
    selected_features = _read_feature_list(feature_list_csv, max_features=max_feature_store_features)
    feature_matrix, feature_store_report = _load_feature_store_columns(
        frame,
        feature_dir=feature_dir,
        selected_features=selected_features,
    )
    for col in feature_matrix.columns:
        frame[col] = feature_matrix[col].to_numpy(dtype=np.float32, copy=False)

    metrics = _path_metrics(frame)
    features = _feature_columns(frame)
    targets = _label_targets(frame, metrics)
    month_period = frame["__ts__"].dt.to_period("M").astype(str)
    train_mask = month_period < month
    valid_mask = month_period == month
    if int(train_mask.sum()) < 100 or int(valid_mask.sum()) < 50:
        raise ValueError(f"Insufficient train/valid rows for month={month}")

    train = frame.loc[train_mask].copy()
    valid = frame.loc[valid_mask].copy().reset_index(drop=True)
    train_metrics = metrics.loc[train_mask].copy()
    valid_metrics = metrics.loc[valid_mask].copy().reset_index(drop=True)
    target_train = targets[label_arm].loc[train_mask].copy()
    target_valid = targets[label_arm].loc[valid_mask].copy().reset_index(drop=True)
    weights = _weight_series(
        frame=train,
        metrics=train_metrics,
        target=target_train,
        arm=weight_arm,
    )
    score, score_diag = _weighted_proxy_score(
        train,
        frame.loc[valid_mask].copy(),
        features,
        target_train["target_soft"],
        weights,
        method=proxy_method,
        tail_frac=tail_recovery_frac,
    )
    score = score.reset_index(drop=True)
    proxy_features = list(score_diag.get("proxy_features", []))

    if oracle_basis == "utility":
        oracle_score = valid_metrics["u_policy_net"]
    elif oracle_basis == "target_soft":
        oracle_score = target_valid["target_soft"]
    else:
        raise ValueError("oracle_basis must be one of: utility,target_soft")

    oracle_mask = _top_mask(oracle_score, top_frac)
    proxy_mask = _top_mask(score, top_frac)
    recovered_mask = oracle_mask & proxy_mask
    missed_mask = oracle_mask & ~proxy_mask
    false_positive_mask = proxy_mask & ~oracle_mask
    neither_mask = ~(oracle_mask | proxy_mask)

    group_rows = [
        _group_metrics(
            frame=valid,
            metrics=valid_metrics,
            score=score,
            target=target_valid,
            mask=mask,
            group=group,
            top_frac=top_frac,
        )
        for group, mask in [
            ("all_valid_month", pd.Series(True, index=valid.index)),
            ("oracle_top", oracle_mask),
            ("proxy_top", proxy_mask),
            ("recovered_winners", recovered_mask),
            ("missed_winners", missed_mask),
            ("false_positives", false_positive_mask),
            ("neither", neither_mask),
        ]
    ]
    group_summary = pd.DataFrame(group_rows)
    feature_contrast = _feature_contrasts(
        frame=valid,
        features=features,
        missed_mask=missed_mask,
        false_positive_mask=false_positive_mask,
        oracle_mask=oracle_mask,
        proxy_mask=proxy_mask,
    )
    ic_stability = _feature_ic_stability(
        train=train,
        valid=valid,
        features=features,
        target_train=target_train["target_soft"],
        target_valid=target_valid["target_soft"],
        utility_train=train_metrics["u_policy_net"],
        utility_valid=valid_metrics["u_policy_net"],
        weights=weights,
        proxy_features=proxy_features,
    )
    missed_winners = _row_extract(
        frame=valid,
        metrics=valid_metrics,
        score=score,
        target=target_valid,
        mask=missed_mask,
        features=proxy_features,
        limit=row_limit,
    )
    false_positives = _row_extract(
        frame=valid,
        metrics=valid_metrics,
        score=score,
        target=target_valid,
        mask=false_positive_mask,
        features=proxy_features,
        limit=row_limit,
    )

    paths = {
        "group_summary": output_dir / "group_summary.csv",
        "feature_contrast": output_dir / "feature_contrast_missed_vs_false_positive.csv",
        "ic_stability": output_dir / "feature_ic_stability.csv",
        "missed_winners": output_dir / "missed_winners.csv",
        "false_positives": output_dir / "false_positives.csv",
        "manifest": output_dir / "manifest.json",
    }
    group_summary.to_csv(paths["group_summary"], index=False)
    feature_contrast.to_csv(paths["feature_contrast"], index=False)
    ic_stability.to_csv(paths["ic_stability"], index=False)
    missed_winners.to_csv(paths["missed_winners"], index=False)
    false_positives.to_csv(paths["false_positives"], index=False)

    manifest = {
        "labels_path": str(labels_path),
        "output_dir": str(output_dir),
        "month": month,
        "label_arm": label_arm,
        "weight_arm": weight_arm,
        "top_frac": float(top_frac),
        "oracle_basis": oracle_basis,
        "proxy_method": proxy_method,
        "tail_recovery_frac": float(tail_recovery_frac),
        "rows": int(len(frame)),
        "train_rows": int(train_mask.sum()),
        "valid_rows": int(valid_mask.sum()),
        "symbols": int(frame["__symbol__"].nunique(dropna=True)),
        "feature_count": int(len(features)),
        "proxy_features": proxy_features,
        "proxy_top_abs_ic": score_diag.get("proxy_top_abs_ic"),
        "proxy_mean_top_abs_ic": score_diag.get("proxy_mean_top_abs_ic"),
        "score_ic_u_valid": _spearman(score, valid_metrics["u_policy_net"]),
        "score_ic_label_valid": _spearman(score, target_valid["target_soft"]),
        "weight_effective_n": _effective_sample_size(weights),
        "weight_effective_frac": _effective_sample_size(weights) / float(len(weights))
        if len(weights)
        else float("nan"),
        "feature_store": feature_store_report,
        "feature_dir": str(feature_dir),
        "feature_list_csv": str(feature_list_csv),
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    markdown = _write_markdown(
        output_dir=output_dir,
        group_summary=group_summary,
        feature_contrast=feature_contrast,
        ic_stability=ic_stability,
        manifest=manifest,
    )
    manifest["outputs"]["markdown"] = str(markdown)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--max-feature-store-features", type=int, default=None)
    parser.add_argument("--month", default="2026-06")
    parser.add_argument("--label-arm", default="S13_grid_mild_time075")
    parser.add_argument("--weight-arm", default="W4_opportunity_miss")
    parser.add_argument("--top-frac", type=float, default=0.05)
    parser.add_argument("--row-limit", type=int, default=100)
    parser.add_argument("--oracle-basis", choices=("utility", "target_soft"), default="utility")
    parser.add_argument("--proxy-method", choices=PROXY_METHODS, default="weighted_ic")
    parser.add_argument("--tail-recovery-frac", type=float, default=0.01)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_diagnostic(
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        max_feature_store_features=args.max_feature_store_features,
        month=str(args.month),
        label_arm=str(args.label_arm),
        weight_arm=str(args.weight_arm),
        top_frac=float(args.top_frac),
        row_limit=int(args.row_limit),
        oracle_basis=str(args.oracle_basis),
        proxy_method=str(args.proxy_method),
        tail_recovery_frac=float(args.tail_recovery_frac),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
