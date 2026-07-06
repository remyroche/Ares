#!/usr/bin/env python3
"""Diagnose feature separability at the blocked GMM train-base gate.

The learnability gate currently stops before train_meta because near-pass
selectors trade off bad-MAE, timeout, and minimum exposure. This diagnostic
replays one OOS month/selector, then compares selected risky rows against clean
rows the selector missed to identify causal feature-store signals that could
support the next target or feature-selection repair.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_label_feature_store_model_smoke import (
    _constrained_top_indices,
    _fit_predict,
    _fit_risk_prediction,
    _fixed_artifact_targets,
    _month_model_frame,
    _score_from_selected_indices,
)
from scripts.run_label_economic_proxy_ablation import _label_targets
from scripts.run_label_quality_proxy_diagnostics import (
    _feature_columns,
    _json_safe,
    _load_feature_store_columns,
    _load_labels,
    _path_metrics,
    _rank_top_indices,
    _read_feature_list,
    _safe_mean,
    _safe_quantile,
    _spearman,
)
from scripts.run_label_weighted_proxy_ablation import _weight_series


DEFAULT_REPORT_DIR = Path("data_perp/reports/gmm_cluster_policy_smoke_20260702_wide_sidebalanced")
DEFAULT_OUTPUT_SUBDIR = "gmm_train_base_feature_gap_diagnostics"
DEFAULT_LABEL_ARM = "OPTIMIZED_ECONOMIC_TIMEOUT_SAFE_TARGET"
DEFAULT_SELECTOR_VARIANT = "strong_bad_mae_timeout_penalty_pred_bad_mae_cap_52_side_cap_70"
GMM_CONTEXT_TOKENS = (
    "gmm",
    "cluster",
    "state",
    "posterior",
    "entropy",
    "reconstruct",
    "spectral",
    "ae_",
    "_ae",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_readiness_row(report_dir: Path) -> dict[str, Any]:
    readiness = pd.read_csv(report_dir / "gmm_train_meta_readiness.csv")
    if readiness.empty:
        raise ValueError("gmm_train_meta_readiness.csv has no active candidate")
    if len(readiness) != 1:
        raise ValueError(f"Expected one active candidate, found {len(readiness)}")
    return readiness.iloc[0].to_dict()


def _parse_bad_mae_cap(selector_variant: str, default: float) -> float:
    match = re.search(r"pred_bad_mae_cap_(\d+)", selector_variant)
    if not match:
        return float(default)
    return float(match.group(1)) / 100.0


def _selected_indices_for_variant(
    *,
    selector_variant: str,
    utility_score: pd.Series,
    bad_mae_pred: pd.Series,
    timeout_pred: pd.Series,
    clean_path_pred: pd.Series,
    side: pd.Series,
    top_frac: float,
    max_side_share: float,
    pred_timeout_cap: float,
) -> tuple[pd.Series, np.ndarray, dict[str, Any]]:
    if selector_variant.startswith("clean_path_probability_min_"):
        match = re.search(r"clean_path_probability_min_(\d+)", selector_variant)
        threshold = float(match.group(1)) / 100.0 if match else 0.35
        eligible = pd.to_numeric(clean_path_pred, errors="coerce") >= threshold
        selected, diag = _constrained_top_indices(
            score=clean_path_pred,
            side=side,
            eligible=eligible,
            top_frac=top_frac,
            max_side_share=max_side_share,
        )
        return (
            _score_from_selected_indices(base_score=clean_path_pred, selected_idx=selected),
            selected,
            {"score_family": "clean_path_probability", "min_clean_path_pred": threshold, **diag},
        )

    strong_score = (utility_score - 0.55 * bad_mae_pred - 0.15 * timeout_pred).astype(np.float32)
    if "pred_bad_mae_cap" in selector_variant:
        cap = _parse_bad_mae_cap(selector_variant, 0.52)
        eligible = (
            pd.to_numeric(bad_mae_pred, errors="coerce").le(cap)
            & pd.to_numeric(timeout_pred, errors="coerce").le(pred_timeout_cap)
        )
        selected, diag = _constrained_top_indices(
            score=strong_score,
            side=side,
            eligible=eligible,
            top_frac=top_frac,
            max_side_share=max_side_share,
        )
        return (
            _score_from_selected_indices(base_score=strong_score, selected_idx=selected),
            selected,
            {
                "score_family": "strong_bad_mae_timeout_penalty",
                "pred_bad_mae_cap": cap,
                "pred_timeout_cap": pred_timeout_cap,
                **diag,
            },
        )

    score = strong_score
    selected = _rank_top_indices(score, top_frac)
    return score, selected, {"score_family": "strong_bad_mae_timeout_penalty"}


def _auc_direction(values: pd.Series, positive_mask: pd.Series, negative_mask: pd.Series) -> float:
    work = pd.DataFrame(
        {
            "value": pd.to_numeric(values, errors="coerce"),
            "label": np.where(positive_mask, 1.0, np.where(negative_mask, 0.0, np.nan)),
        }
    ).dropna()
    if len(work) < 5 or work["label"].nunique() < 2 or work["value"].nunique() < 2:
        return float("nan")
    ranks = work["value"].rank(method="average")
    pos = work["label"].eq(1.0)
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    u_stat = float(ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0)
    return u_stat / float(n_pos * n_neg)


def _feature_gap_table(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    features: list[str],
    selected_mask: pd.Series,
) -> pd.DataFrame:
    clean = (
        pd.to_numeric(metrics["u_policy_net"], errors="coerce").gt(0.0)
        & pd.to_numeric(metrics["mae_norm"], errors="coerce").lt(1.0)
        & metrics["is_timeout"].astype(float).le(0.0)
    )
    bad_mae = pd.to_numeric(metrics["mae_norm"], errors="coerce").ge(1.0)
    timeout = metrics["is_timeout"].astype(float).gt(0.0)
    risky_selected = selected_mask & (~clean)
    rejected_clean = (~selected_mask) & clean

    month_period = pd.to_datetime(frame["__ts__"], errors="coerce").dt.to_period("M").astype(str)
    rows: list[dict[str, Any]] = []
    for feature in features:
        values = pd.to_numeric(frame[feature], errors="coerce")
        selected_values = values[risky_selected]
        clean_values = values[rejected_clean]
        if int(selected_values.notna().sum()) < 3 or int(clean_values.notna().sum()) < 10:
            continue
        selected_median = _safe_quantile(selected_values, 0.50)
        clean_median = _safe_quantile(clean_values, 0.50)
        pooled = pd.concat([selected_values, clean_values]).dropna()
        iqr = float(pooled.quantile(0.75) - pooled.quantile(0.25)) if len(pooled) else float("nan")
        robust_delta = (
            (selected_median - clean_median) / iqr
            if math.isfinite(selected_median)
            and math.isfinite(clean_median)
            and math.isfinite(iqr)
            and abs(iqr) > 1e-12
            else float("nan")
        )
        clean_ics: list[float] = []
        bad_ics: list[float] = []
        timeout_ics: list[float] = []
        for _month, ids in values.groupby(month_period, dropna=False).groups.items():
            idx = pd.Index(ids)
            if len(idx) < 100:
                continue
            clean_ics.append(_spearman(values.loc[idx], clean.loc[idx].astype(float)))
            bad_ics.append(_spearman(values.loc[idx], bad_mae.loc[idx].astype(float)))
            timeout_ics.append(_spearman(values.loc[idx], timeout.loc[idx].astype(float)))
        finite_clean_ics = [v for v in clean_ics if math.isfinite(v)]
        global_clean_ic = _spearman(values, clean.astype(float))
        global_bad_ic = _spearman(values, bad_mae.astype(float))
        global_timeout_ic = _spearman(values, timeout.astype(float))
        sign = 1.0 if global_clean_ic >= 0.0 else -1.0
        sign_stability = (
            float(np.mean([np.sign(v) == sign for v in finite_clean_ics]))
            if finite_clean_ics
            else float("nan")
        )
        auc_clean_vs_risky = _auc_direction(values, rejected_clean, risky_selected)
        feature_lower = feature.lower()
        rows.append(
            {
                "feature": feature,
                "selected_risky_median": selected_median,
                "rejected_clean_median": clean_median,
                "selected_minus_rejected_clean_delta": selected_median - clean_median,
                "selected_minus_rejected_clean_robust_delta": robust_delta,
                "auc_rejected_clean_vs_selected_risky": auc_clean_vs_risky,
                "global_clean_ic": global_clean_ic,
                "global_bad_mae_ic": global_bad_ic,
                "global_timeout_ic": global_timeout_ic,
                "month_clean_ic_mean": _safe_mean(finite_clean_ics),
                "month_bad_mae_ic_mean": _safe_mean([v for v in bad_ics if math.isfinite(v)]),
                "month_timeout_ic_mean": _safe_mean([v for v in timeout_ics if math.isfinite(v)]),
                "clean_sign_stability": sign_stability,
                "month_count": int(len(finite_clean_ics)),
                "gmm_context_feature": any(token in feature_lower for token in GMM_CONTEXT_TOKENS),
                "interpretability_score": (
                    abs(robust_delta) if math.isfinite(robust_delta) else 0.0
                )
                + 2.0 * abs(global_clean_ic if math.isfinite(global_clean_ic) else 0.0)
                + 0.50 * (sign_stability if math.isfinite(sign_stability) else 0.0),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["interpretability_score", "clean_sign_stability", "global_clean_ic"],
        ascending=[False, False, False],
    )


def run_diagnostic(
    *,
    report_dir: Path,
    output_dir: Path,
    label_arm: str,
    selector_variant: str,
    month: str,
    pred_timeout_cap: float,
) -> dict[str, Any]:
    readiness = _read_readiness_row(report_dir)
    report_manifest = _read_json(report_dir / "manifest.json")
    seeds = [int(v) for v in report_manifest.get("seeds", [17, 29])]
    train_lookback_months = int(report_manifest.get("train_lookback_months", 2))
    top_frac = float(readiness.get("top_frac", 0.03))
    max_side_share = 0.70

    frame = _load_labels(Path(str(readiness["labels_path"])))
    selected_features = _read_feature_list(Path(str(readiness["feature_list_csv"])))
    feature_matrix, feature_store_report = _load_feature_store_columns(
        frame,
        feature_dir=Path(str(readiness["feature_dir"])),
        selected_features=selected_features,
    )
    if not feature_matrix.empty:
        frame = pd.concat(
            [frame.reset_index(drop=True), feature_matrix.astype(np.float32, copy=False).reset_index(drop=True)],
            axis=1,
            copy=False,
        )
    metrics = _path_metrics(frame)
    eval_col = str(readiness.get("evaluation_utility_source") or "").strip()
    if eval_col:
        metrics["u_policy_net"] = pd.to_numeric(frame[eval_col], errors="coerce").astype(np.float32)
    targets = _label_targets(frame, metrics)
    targets.update(_fixed_artifact_targets(frame, metrics))
    if label_arm not in targets:
        raise ValueError(f"Unknown label arm {label_arm}. Available: {sorted(targets)}")
    features = _feature_columns(frame)
    month_period = frame["__ts__"].dt.to_period("M").astype(str)
    train_mask = month_period < month
    if train_lookback_months > 0:
        prior_months = sorted(month_period[train_mask].dropna().unique())
        train_mask = train_mask & month_period.isin(set(prior_months[-train_lookback_months:]))
    valid_mask = month_period == month
    if int(train_mask.sum()) < 500 or int(valid_mask.sum()) < 100:
        raise ValueError(
            f"Insufficient rows for {month}: train={int(train_mask.sum())} valid={int(valid_mask.sum())}"
        )

    x_train, x_valid = _month_model_frame(
        frame,
        train_mask=train_mask,
        valid_mask=valid_mask,
        features=features,
    )
    train = frame.loc[train_mask].copy()
    valid = frame.loc[valid_mask].copy().reset_index(drop=True)
    train_metrics = metrics.loc[train_mask].copy().reset_index(drop=True)
    valid_metrics = metrics.loc[valid_mask].copy().reset_index(drop=True)
    target_train = targets[label_arm].loc[train_mask].copy().reset_index(drop=True)
    weights = _weight_series(
        frame=train.reset_index(drop=True),
        metrics=train_metrics,
        target=target_train,
        arm=str(readiness.get("weight_arm", "W0_base")),
    )
    utility_preds = [
        _fit_predict(
            x_train=x_train,
            y_train=target_train["target_soft"],
            w_train=weights,
            x_valid=x_valid,
            seed=seed,
        )
        for seed in seeds
    ]
    utility_score = pd.Series(np.mean(np.vstack(utility_preds), axis=0).astype(np.float32))
    bad_mae_train = (train_metrics["mae_norm"] >= 1.0).astype(float)
    timeout_train = train_metrics["is_timeout"].astype(float)
    clean_path_train = (
        (train_metrics["u_policy_net"] > 0.0)
        & (train_metrics["mae_norm"] < 1.0)
        & (train_metrics["is_timeout"].astype(float) <= 0.0)
    ).astype(float)
    bad_mae_pred = pd.Series(
        _fit_risk_prediction(
            x_train=x_train,
            y_train=bad_mae_train,
            x_valid=x_valid,
            seeds=seeds,
        )
    )
    timeout_pred = pd.Series(
        _fit_risk_prediction(
            x_train=x_train,
            y_train=timeout_train,
            x_valid=x_valid,
            seeds=seeds,
        )
    )
    clean_path_pred = pd.Series(
        _fit_risk_prediction(
            x_train=x_train,
            y_train=clean_path_train,
            x_valid=x_valid,
            seeds=[seed + 50_000 for seed in seeds],
        )
    )
    final_score, selected_idx, selector_diag = _selected_indices_for_variant(
        selector_variant=selector_variant,
        utility_score=utility_score,
        bad_mae_pred=bad_mae_pred,
        timeout_pred=timeout_pred,
        clean_path_pred=clean_path_pred,
        side=valid_metrics["side"],
        top_frac=top_frac,
        max_side_share=max_side_share,
        pred_timeout_cap=pred_timeout_cap,
    )
    selected_mask = pd.Series(False, index=valid.index)
    selected_mask.iloc[selected_idx] = True
    clean = (
        (valid_metrics["u_policy_net"] > 0.0)
        & (valid_metrics["mae_norm"] < 1.0)
        & (valid_metrics["is_timeout"].astype(float) <= 0.0)
    )
    feature_gap = _feature_gap_table(
        frame=valid,
        metrics=valid_metrics,
        features=features,
        selected_mask=selected_mask,
    )

    ledger = pd.DataFrame(
        {
            "__ts__": valid["__ts__"],
            "__symbol__": valid["__symbol__"],
            "side": valid_metrics["side"],
            "selected": selected_mask,
            "clean_positive": clean,
            "u": valid_metrics["u_policy_net"],
            "mae_norm": valid_metrics["mae_norm"],
            "timeout": valid_metrics["is_timeout"].astype(float),
            "utility_score": utility_score,
            "bad_mae_pred": bad_mae_pred,
            "timeout_pred": timeout_pred,
            "clean_path_pred": clean_path_pred,
            "final_score": final_score,
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_gap_path = output_dir / "gmm_train_base_feature_gap_features.csv"
    ledger_path = output_dir / "gmm_train_base_feature_gap_ledger.csv"
    manifest_path = output_dir / "gmm_train_base_feature_gap_manifest.json"
    feature_gap.to_csv(feature_gap_path, index=False)
    ledger.to_csv(ledger_path, index=False)
    selected_metrics = valid_metrics.loc[selected_mask].copy()
    summary = {
        "status": "diagnostic_only",
        "report_dir": str(report_dir),
        "label_arm": label_arm,
        "selector_variant": selector_variant,
        "month": month,
        "train_rows": int(train_mask.sum()),
        "valid_rows": int(valid_mask.sum()),
        "selected_rows": int(selected_mask.sum()),
        "rejected_clean_rows": int(((~selected_mask) & clean).sum()),
        "selected_mean_u": _safe_mean(selected_metrics["u_policy_net"]),
        "selected_bad_mae_1r_rate": _safe_mean(selected_metrics["mae_norm"] >= 1.0),
        "selected_timeout_rate": _safe_mean(selected_metrics["is_timeout"].astype(float)),
        "selected_long_share": _safe_mean(selected_metrics["side"] > 0.0),
        "selected_short_share": _safe_mean(selected_metrics["side"] < 0.0),
        "score_ic_u": _spearman(final_score, valid_metrics["u_policy_net"]),
        "score_ic_bad_mae": _spearman(final_score, (valid_metrics["mae_norm"] >= 1.0).astype(float)),
        "score_ic_timeout": _spearman(final_score, valid_metrics["is_timeout"].astype(float)),
        "selector_diag": selector_diag,
        "feature_store": feature_store_report,
        "top_feature_candidates": feature_gap.head(25).to_dict(orient="records"),
        "outputs": {
            "features": str(feature_gap_path),
            "ledger": str(ledger_path),
            "manifest": str(manifest_path),
        },
    }
    manifest_path.write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--label-arm", type=str, default=DEFAULT_LABEL_ARM)
    parser.add_argument("--selector-variant", type=str, default=DEFAULT_SELECTOR_VARIANT)
    parser.add_argument("--month", type=str, default="2026-06")
    parser.add_argument("--pred-timeout-cap", type=float, default=0.12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or (args.report_dir / DEFAULT_OUTPUT_SUBDIR)
    summary = run_diagnostic(
        report_dir=args.report_dir,
        output_dir=output_dir,
        label_arm=args.label_arm,
        selector_variant=args.selector_variant,
        month=args.month,
        pred_timeout_cap=float(args.pred_timeout_cap),
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
