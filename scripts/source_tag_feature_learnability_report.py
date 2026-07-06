#!/usr/bin/env python3
"""Diagnose whether source-aware quality labels are learnable by current features.

This is diagnostic-only. It reuses the same month-forward ExtraTrees smoke
setup as the v17 source-quality ablation reports, then breaks the result down
by primary source tag. For each ablation/month/source bucket it reports:

* target/utility alignment;
* model score alignment to the label and realized utility;
* train-period feature IC versus validation-period feature IC;
* sign stability of the strongest train-period feature signals.

No production training artifacts are written.
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

from scripts.run_label_feature_store_model_smoke import _fit_predict, _month_model_frame  # noqa: E402
from scripts.run_label_quality_proxy_diagnostics import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    DEFAULT_FEATURE_LIST_CSV,
    DEFAULT_LABELS_DIR,
    _json_safe,
    _load_feature_store_columns,
    _make_targets,
    _path_metrics,
    _read_feature_list,
    _safe_mean,
    _safe_quantile,
    _spearman,
)
from scripts.run_source_quality_label_walkforward_ablation import (  # noqa: E402
    DEFAULT_ABLATIONS,
    DEFAULT_MANIFEST,
    DEFAULT_MONTHS,
    DEFAULT_QUALITY_LABELS,
    DEFAULT_SEEDS,
    VANILLA_LABEL_ARM,
    VANILLA_NAME,
    _load_joined_frame,
    _load_manifest_specs,
    _parse_csv,
    _parse_int_csv,
    _rank_top_indices_local,
    _source_feature_columns,
    _target_for_spec,
    _training_mask_for_spec,
    _weight_series,
)


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/source_quality_label_walkforward_ablation_v1_sideaware_20260702/feature_learnability")
DEFAULT_TOP_FRAC = 0.10
DEFAULT_TOP_FEATURES = 12


def _safe_numeric(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _feature_family(feature: str, source_features: set[str]) -> str:
    if feature in source_features or feature.startswith("tag_") or feature.endswith("_score"):
        return "source"
    lowered = feature.lower()
    if "volume" in lowered or "turnover" in lowered:
        return "volume"
    if "spread" in lowered or "slip" in lowered or "liq" in lowered:
        return "execution"
    if "trend" in lowered or "mom" in lowered or "return" in lowered:
        return "trend"
    if "vol" in lowered or "atr" in lowered or "range" in lowered:
        return "volatility"
    if "oi" in lowered or "funding" in lowered:
        return "oi"
    return "base"


def _bucket_masks(frame: pd.DataFrame, mask: pd.Series) -> list[tuple[str, pd.Series]]:
    masks: list[tuple[str, pd.Series]] = [("all_rows", mask.copy())]
    if "primary_source_tag" not in frame.columns:
        return masks
    tags = sorted(str(v) for v in frame.loc[mask, "primary_source_tag"].dropna().astype(str).unique())
    for tag in tags:
        masks.append((tag, mask & frame["primary_source_tag"].astype(str).eq(tag)))
    return masks


def _top_mean(score: pd.Series, utility: pd.Series, frac: float) -> float:
    idx = _rank_top_indices_local(score.reset_index(drop=True), frac)
    if len(idx) == 0:
        return float("nan")
    return _safe_mean(utility.reset_index(drop=True).iloc[idx])


def _top_hit(score: pd.Series, utility: pd.Series, frac: float) -> float:
    idx = _rank_top_indices_local(score.reset_index(drop=True), frac)
    if len(idx) == 0:
        return float("nan")
    return _safe_mean(utility.reset_index(drop=True).iloc[idx] > 0.0)


def _quality_distribution(frame: pd.DataFrame, label_column: str, mask: pd.Series) -> dict[str, Any]:
    if label_column not in frame.columns:
        return {
            "quality_good_rows": 0,
            "quality_bad_rows": 0,
            "quality_neutral_rows": 0,
            "quality_good_rate": float("nan"),
            "quality_bad_rate": float("nan"),
            "quality_neutral_rate": float("nan"),
        }
    label = _safe_numeric(frame.loc[mask, label_column])
    rows = int(label.notna().sum())
    good = int(label.eq(1.0).sum())
    bad = int(label.eq(0.0).sum())
    neutral = int(label.eq(-1.0).sum())
    denom = rows if rows else 0
    return {
        "quality_good_rows": good,
        "quality_bad_rows": bad,
        "quality_neutral_rows": neutral,
        "quality_good_rate": float(good / denom) if denom else float("nan"),
        "quality_bad_rate": float(bad / denom) if denom else float("nan"),
        "quality_neutral_rate": float(neutral / denom) if denom else float("nan"),
    }


def _feature_ic_frame(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    target: pd.DataFrame,
    features: list[str],
    mask: pd.Series,
    source_features: set[str],
    context: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if int(mask.sum()) < 10:
        return pd.DataFrame()
    target_soft = target.loc[mask, "target_soft"]
    utility = metrics.loc[mask, "u_policy_net"]
    for feature in features:
        values = frame.loc[mask, feature]
        finite = values.notna() & target_soft.notna() & utility.notna()
        finite_rows = int(finite.sum())
        if finite_rows < 10:
            continue
        ic_label = _spearman(values.loc[finite], target_soft.loc[finite])
        ic_u = _spearman(values.loc[finite], utility.loc[finite])
        if not math.isfinite(ic_label) and not math.isfinite(ic_u):
            continue
        rows.append(
            {
                **context,
                "feature": feature,
                "feature_family": _feature_family(feature, source_features),
                "finite_rows": finite_rows,
                "ic_label": ic_label,
                "abs_ic_label": abs(ic_label) if math.isfinite(ic_label) else float("nan"),
                "ic_u": ic_u,
                "abs_ic_u": abs(ic_u) if math.isfinite(ic_u) else float("nan"),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["abs_ic_label", "abs_ic_u"], ascending=[False, False], na_position="last")


def _top_feature_text(feature_ic: pd.DataFrame, metric: str, limit: int) -> str:
    if feature_ic.empty or metric not in feature_ic.columns:
        return ""
    view = feature_ic[np.isfinite(pd.to_numeric(feature_ic[metric], errors="coerce"))].copy()
    if view.empty:
        return ""
    view["abs_metric"] = pd.to_numeric(view[metric], errors="coerce").abs()
    view = view.sort_values("abs_metric", ascending=False).head(limit)
    parts = []
    for _, row in view.iterrows():
        parts.append(f"{row['feature']}:{float(row[metric]):+.3f}")
    return "; ".join(parts)


def _signal_summary(train_ic: pd.DataFrame, valid_ic: pd.DataFrame, *, top_features: int) -> dict[str, Any]:
    if train_ic.empty:
        return {
            "train_feature_count": 0,
            "train_max_abs_ic_label": float("nan"),
            "train_top_feature_mean_abs_ic_label": float("nan"),
            "valid_max_abs_ic_label": float("nan"),
            "valid_max_abs_ic_u": float("nan"),
            "top_feature_label_sign_agreement": float("nan"),
            "top_feature_valid_label_utility_sign_agreement": float("nan"),
            "top_train_label_features": "",
            "top_valid_label_features": _top_feature_text(valid_ic, "ic_label", top_features),
            "top_valid_utility_features": _top_feature_text(valid_ic, "ic_u", top_features),
        }
    train_sorted = train_ic.sort_values("abs_ic_label", ascending=False, na_position="last").head(top_features)
    valid_lookup = valid_ic.set_index("feature") if not valid_ic.empty else pd.DataFrame()
    agreements: list[float] = []
    utility_agreements: list[float] = []
    for _, row in train_sorted.iterrows():
        feature = str(row["feature"])
        train_label_ic = float(row["ic_label"]) if pd.notna(row["ic_label"]) else float("nan")
        if not math.isfinite(train_label_ic) or abs(train_label_ic) <= 0.0 or valid_lookup.empty or feature not in valid_lookup.index:
            continue
        valid_label_ic = float(valid_lookup.loc[feature, "ic_label"])
        valid_u_ic = float(valid_lookup.loc[feature, "ic_u"])
        if math.isfinite(valid_label_ic) and abs(valid_label_ic) > 0.0:
            agreements.append(float(np.sign(train_label_ic) == np.sign(valid_label_ic)))
        if math.isfinite(valid_u_ic) and abs(valid_u_ic) > 0.0:
            utility_agreements.append(float(np.sign(train_label_ic) == np.sign(valid_u_ic)))
    return {
        "train_feature_count": int(train_ic["feature"].nunique()),
        "train_max_abs_ic_label": _safe_quantile(train_ic["abs_ic_label"], 1.0),
        "train_top_feature_mean_abs_ic_label": _safe_mean(train_sorted["abs_ic_label"]),
        "valid_max_abs_ic_label": _safe_quantile(valid_ic["abs_ic_label"], 1.0) if not valid_ic.empty else float("nan"),
        "valid_max_abs_ic_u": _safe_quantile(valid_ic["abs_ic_u"], 1.0) if not valid_ic.empty else float("nan"),
        "top_feature_label_sign_agreement": _safe_mean(agreements),
        "top_feature_valid_label_utility_sign_agreement": _safe_mean(utility_agreements),
        "top_train_label_features": _top_feature_text(train_ic, "ic_label", top_features),
        "top_valid_label_features": _top_feature_text(valid_ic, "ic_label", top_features),
        "top_valid_utility_features": _top_feature_text(valid_ic, "ic_u", top_features),
    }


def _diagnose(row: dict[str, Any]) -> str:
    valid_rows = int(row.get("valid_rows", 0) or 0)
    if valid_rows < int(row.get("min_bucket_rows", 0) or 0):
        return "too_sparse"
    target_ic_u = float(row.get("target_ic_u", float("nan")))
    oracle_top_u = float(row.get("target_oracle_top_mean_u", float("nan")))
    model_ic_label = float(row.get("model_ic_label", float("nan")))
    model_ic_u = float(row.get("model_ic_u", float("nan")))
    model_top_u = float(row.get("model_top_mean_u", float("nan")))
    train_abs = float(row.get("train_top_feature_mean_abs_ic_label", float("nan")))
    sign_agree = float(row.get("top_feature_label_sign_agreement", float("nan")))
    if math.isfinite(target_ic_u) and target_ic_u <= 0.0 and math.isfinite(oracle_top_u) and oracle_top_u <= 0.0:
        return "label_not_economic_in_bucket"
    if math.isfinite(target_ic_u) and target_ic_u > 0.0:
        if math.isfinite(model_ic_label) and model_ic_label < 0.0:
            return "model_anti_learns_label"
        if math.isfinite(model_ic_u) and model_ic_u < 0.0 and math.isfinite(model_ic_label) and model_ic_label > 0.0:
            return "model_learns_label_but_not_utility"
        if math.isfinite(train_abs) and train_abs < 0.03:
            return "feature_signal_too_weak"
        if math.isfinite(sign_agree) and sign_agree < 0.50:
            return "feature_sign_instability"
        if math.isfinite(oracle_top_u) and oracle_top_u > 0.0 and math.isfinite(model_top_u) and model_top_u <= 0.0:
            return "model_selection_failure"
        if math.isfinite(model_ic_u) and model_ic_u > 0.0 and math.isfinite(model_top_u) and model_top_u > 0.0:
            return "promising_bucket"
    return "weak_or_mixed_signal"


def _recommend(group: pd.DataFrame, *, expected_months: int, min_bucket_rows: int) -> str:
    months = int(group["period"].nunique())
    if months < min(2, expected_months):
        return "too_little_evidence"
    valid_rows_mean = _safe_mean(group["valid_rows"])
    if math.isfinite(valid_rows_mean) and valid_rows_mean < min_bucket_rows:
        return "too_sparse"
    target_positive = int((_safe_numeric(group["target_ic_u"]) > 0.0).sum())
    oracle_positive = int((_safe_numeric(group["target_oracle_top_mean_u"]) > 0.0).sum())
    model_positive = int((_safe_numeric(group["model_top_mean_u"]) > 0.0).sum())
    model_ic_positive = int((_safe_numeric(group["model_ic_u"]) > 0.0).sum())
    sign_stable = int((_safe_numeric(group["top_feature_label_sign_agreement"]) >= 0.50).sum())
    train_signal = _safe_mean(group["train_top_feature_mean_abs_ic_label"])
    mean_model_top_u = _safe_mean(group["model_top_mean_u"])
    mean_model_ic_u = _safe_mean(group["model_ic_u"])
    if (
        target_positive >= expected_months
        and oracle_positive >= expected_months
        and model_positive >= expected_months
        and model_ic_positive >= expected_months
        and math.isfinite(mean_model_top_u)
        and mean_model_top_u > 0.0
        and math.isfinite(mean_model_ic_u)
        and mean_model_ic_u > 0.0
    ):
        return "candidate_training_signal"
    if target_positive >= 2 and oracle_positive >= 2 and math.isfinite(train_signal) and train_signal < 0.03:
        return "needs_better_features_or_simpler_label"
    if target_positive >= 2 and oracle_positive >= 2 and sign_stable < 2:
        return "feature_sign_instability"
    if target_positive >= 2 and oracle_positive >= 2:
        return "model_or_target_rework_before_training"
    return "diagnostic_only"


def _aggregate(summary: pd.DataFrame, *, expected_months: int, min_bucket_rows: int) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for key, group in summary.groupby(["ablation", "source_bucket"], dropna=False, observed=True):
        ablation, bucket = key
        diagnoses = group["diagnosis"].dropna().astype(str)
        rows.append(
            {
                "ablation": ablation,
                "source_bucket": bucket,
                "months": int(group["period"].nunique()),
                "mean_valid_rows": _safe_mean(group["valid_rows"]),
                "total_valid_rows": int(_safe_numeric(group["valid_rows"]).sum()),
                "target_ic_positive_months": int((_safe_numeric(group["target_ic_u"]) > 0.0).sum()),
                "oracle_top_positive_months": int((_safe_numeric(group["target_oracle_top_mean_u"]) > 0.0).sum()),
                "model_top_positive_months": int((_safe_numeric(group["model_top_mean_u"]) > 0.0).sum()),
                "model_ic_u_positive_months": int((_safe_numeric(group["model_ic_u"]) > 0.0).sum()),
                "mean_target_ic_u": _safe_mean(group["target_ic_u"]),
                "mean_model_ic_label": _safe_mean(group["model_ic_label"]),
                "mean_model_ic_u": _safe_mean(group["model_ic_u"]),
                "mean_oracle_top_u": _safe_mean(group["target_oracle_top_mean_u"]),
                "mean_model_top_u": _safe_mean(group["model_top_mean_u"]),
                "mean_model_vs_oracle_top_u_gap": _safe_mean(group["model_vs_oracle_top_mean_u_gap"]),
                "mean_train_top_abs_ic_label": _safe_mean(group["train_top_feature_mean_abs_ic_label"]),
                "mean_valid_max_abs_ic_label": _safe_mean(group["valid_max_abs_ic_label"]),
                "mean_valid_max_abs_ic_u": _safe_mean(group["valid_max_abs_ic_u"]),
                "mean_top_feature_label_sign_agreement": _safe_mean(group["top_feature_label_sign_agreement"]),
                "mean_top_feature_valid_label_utility_sign_agreement": _safe_mean(
                    group["top_feature_valid_label_utility_sign_agreement"]
                ),
                "mean_bad_mae_1r_rate": _safe_mean(group["bad_mae_1r_rate"]),
                "mean_timeout_rate": _safe_mean(group["timeout_rate"]),
                "mean_wide_barrier_25bps_rate": _safe_mean(group["wide_barrier_25bps_rate"]),
                "dominant_diagnosis": diagnoses.mode().iloc[0] if len(diagnoses.mode()) else "",
                "recommendation": _recommend(group, expected_months=expected_months, min_bucket_rows=min_bucket_rows),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["recommendation", "mean_model_top_u", "mean_oracle_top_u"],
        ascending=[True, False, False],
        na_position="last",
    )


def _table(frame: pd.DataFrame, cols: list[str], limit: int | None = None) -> str:
    if frame.empty:
        return "No rows."
    view = frame[[col for col in cols if col in frame.columns]].copy()
    if limit is not None:
        view = view.head(limit)
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda value: f"{float(value):.4f}" if pd.notna(value) else "")
    return view.to_markdown(index=False)


def _write_report(output_dir: Path, summary: pd.DataFrame, aggregate: pd.DataFrame, manifest: dict[str, Any]) -> Path:
    path = output_dir / "source_tag_feature_learnability_report.md"
    all_rows = aggregate[aggregate["source_bucket"].eq("all_rows")].copy()
    buckets = aggregate[~aggregate["source_bucket"].eq("all_rows")].copy()
    focus_cols = [
        "recommendation",
        "ablation",
        "source_bucket",
        "months",
        "mean_valid_rows",
        "target_ic_positive_months",
        "oracle_top_positive_months",
        "model_top_positive_months",
        "mean_target_ic_u",
        "mean_model_ic_label",
        "mean_model_ic_u",
        "mean_oracle_top_u",
        "mean_model_top_u",
        "mean_model_vs_oracle_top_u_gap",
        "mean_train_top_abs_ic_label",
        "mean_top_feature_label_sign_agreement",
        "dominant_diagnosis",
    ]
    detail_cols = [
        "period",
        "ablation",
        "source_bucket",
        "valid_rows",
        "target_ic_u",
        "model_ic_label",
        "model_ic_u",
        "target_oracle_top_mean_u",
        "model_top_mean_u",
        "model_vs_oracle_top_mean_u_gap",
        "train_top_feature_mean_abs_ic_label",
        "top_feature_label_sign_agreement",
        "diagnosis",
        "top_train_label_features",
    ]
    lines = [
        "# Source Tag Feature Learnability Report",
        "",
        "Scope: diagnostic month-forward feature/label learnability by source bucket. This is not production training.",
        "",
        "## Alignment",
        "",
        f"- Joined rows: `{manifest['join_report']['joined_rows']}`",
        f"- Utility source: `{manifest.get('utility_source', '')}`",
        f"- Months: `{', '.join(manifest['months'])}`",
        f"- Base feature count: `{manifest['base_feature_count']}`",
        f"- Source feature count: `{manifest['source_feature_count']}`",
        f"- Top selection fraction: `{manifest['top_frac']}`",
        "",
        "## All-Rows Ablation View",
        "",
        _table(all_rows, focus_cols, limit=80),
        "",
        "## Source Bucket View",
        "",
        _table(buckets, focus_cols, limit=120),
        "",
        "## Monthly Detail",
        "",
        _table(summary.sort_values(["period", "ablation", "source_bucket"]), detail_cols, limit=160),
        "",
        "## Outputs",
        "",
        f"- Summary: `{manifest['outputs']['summary']}`",
        f"- Aggregate: `{manifest['outputs']['aggregate']}`",
        f"- Feature IC: `{manifest['outputs']['feature_ic']}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_report(
    *,
    quality_labels_path: Path,
    labels_path: Path,
    manifest_path: Path,
    output_dir: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    max_feature_store_features: int | None,
    ablations: list[str],
    months: list[str],
    seeds: list[int],
    train_lookback_months: int | None,
    min_train_rows: int,
    min_valid_rows: int,
    min_bucket_rows: int,
    top_frac: float,
    top_features: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = _load_manifest_specs(manifest_path, ablations)
    frame, join_report = _load_joined_frame(quality_labels_path=quality_labels_path, labels_path=labels_path)
    selected_features = _read_feature_list(feature_list_csv, max_features=max_feature_store_features)
    feature_matrix, feature_report = _load_feature_store_columns(
        frame,
        feature_dir=feature_dir,
        selected_features=selected_features,
    )
    for col in feature_matrix.columns:
        frame[col] = feature_matrix[col].to_numpy(dtype=np.float32, copy=False)
    metrics = _path_metrics(frame)
    vanilla_targets = _make_targets(frame, metrics)
    base_features = list(feature_matrix.columns)
    source_features = _source_feature_columns(frame)
    source_feature_set = set(source_features)

    summary_rows: list[dict[str, Any]] = []
    feature_ic_parts: list[pd.DataFrame] = []
    month_period = frame["__ts__"].dt.to_period("M").astype(str)

    for month in months:
        valid_mask = month_period.eq(month)
        if int(valid_mask.sum()) < int(min_valid_rows):
            continue
        for spec in specs:
            train_mask = _training_mask_for_spec(
                spec=spec,
                frame=frame,
                month=month,
                train_lookback_months=train_lookback_months,
            )
            if int(train_mask.sum()) < int(min_train_rows):
                summary_rows.append(
                    {
                        "period": month,
                        "ablation": spec.name,
                        "source_bucket": "all_rows",
                        "train_rows": int(train_mask.sum()),
                        "valid_rows": int(valid_mask.sum()),
                        "min_bucket_rows": int(min_bucket_rows),
                        "diagnosis": "too_few_train_rows",
                    }
                )
                continue

            features = list(base_features)
            if spec.add_source_features:
                features = list(dict.fromkeys(features + source_features))
            target = _target_for_spec(spec=spec, frame=frame, vanilla_targets=vanilla_targets)
            x_train, x_valid = _month_model_frame(
                frame,
                train_mask=train_mask,
                valid_mask=valid_mask,
                features=features,
            )
            target_train = target.loc[train_mask]
            if spec.is_vanilla:
                weights = _safe_numeric(frame.loc[train_mask, "__w__"] if "__w__" in frame.columns else 1.0).fillna(1.0)
            else:
                weights = _weight_series(frame, spec.sample_weight_column).loc[train_mask]
            pred_matrix = np.vstack(
                [
                    _fit_predict(
                        x_train=x_train,
                        y_train=target_train["target_soft"],
                        w_train=weights,
                        x_valid=x_valid,
                        seed=seed,
                    )
                    for seed in seeds
                ]
            )
            model_score_valid = pd.Series(
                np.mean(pred_matrix, axis=0).astype(np.float32),
                index=frame.loc[valid_mask].index,
            )
            model_score = pd.Series(np.nan, index=frame.index, dtype=np.float32)
            model_score.loc[valid_mask] = model_score_valid

            train_buckets = dict(_bucket_masks(frame, train_mask))
            for source_bucket, valid_bucket_mask in _bucket_masks(frame, valid_mask):
                train_bucket_mask = train_buckets.get(
                    source_bucket,
                    pd.Series(False, index=frame.index),
                )
                train_rows = int(train_bucket_mask.sum())
                valid_rows = int(valid_bucket_mask.sum())
                if valid_rows == 0:
                    continue
                train_ic = _feature_ic_frame(
                    frame=frame,
                    metrics=metrics,
                    target=target,
                    features=features,
                    mask=train_bucket_mask,
                    source_features=source_feature_set,
                    context={
                        "period": month,
                        "ablation": spec.name,
                        "source_bucket": source_bucket,
                        "split": "train",
                    },
                )
                valid_ic = _feature_ic_frame(
                    frame=frame,
                    metrics=metrics,
                    target=target,
                    features=features,
                    mask=valid_bucket_mask,
                    source_features=source_feature_set,
                    context={
                        "period": month,
                        "ablation": spec.name,
                        "source_bucket": source_bucket,
                        "split": "valid",
                    },
                )
                if not train_ic.empty:
                    feature_ic_parts.append(train_ic)
                if not valid_ic.empty:
                    feature_ic_parts.append(valid_ic)
                signal = _signal_summary(train_ic, valid_ic, top_features=top_features)

                valid_target = target.loc[valid_bucket_mask]
                valid_metrics = metrics.loc[valid_bucket_mask]
                valid_score = model_score.loc[valid_bucket_mask]
                row: dict[str, Any] = {
                    "period": month,
                    "ablation": spec.name,
                    "source_bucket": source_bucket,
                    "label_column": spec.label_column,
                    "sample_weight_column": spec.sample_weight_column,
                    "source_features_added": bool(spec.add_source_features),
                    "model_feature_count": int(len(features)),
                    "train_rows": train_rows,
                    "valid_rows": valid_rows,
                    "min_bucket_rows": int(min_bucket_rows),
                    "target_train_mean": _safe_mean(target.loc[train_bucket_mask, "target_soft"]),
                    "target_valid_mean": _safe_mean(valid_target["target_soft"]),
                    "target_valid_hard_rate": _safe_mean(valid_target["target_hard"] > 0.5),
                    "mean_u": _safe_mean(valid_metrics["u_policy_net"]),
                    "hit_u": _safe_mean(valid_metrics["u_policy_net"] > 0.0),
                    "p25_u": _safe_quantile(valid_metrics["u_policy_net"], 0.25),
                    "bad_mae_1r_rate": _safe_mean(valid_metrics["mae_norm"] >= 1.0),
                    "p90_mae_norm": _safe_quantile(valid_metrics["mae_norm"], 0.90),
                    "timeout_rate": _safe_mean(valid_metrics["is_timeout"].astype(float)),
                    "wide_barrier_25bps_rate": _safe_mean(valid_metrics["barrier"] > 0.025),
                    "target_ic_u": _spearman(valid_target["target_soft"], valid_metrics["u_policy_net"]),
                    "model_ic_label": _spearman(valid_score, valid_target["target_soft"]),
                    "model_ic_u": _spearman(valid_score, valid_metrics["u_policy_net"]),
                    "target_oracle_top_mean_u": _top_mean(
                        valid_target["target_soft"],
                        valid_metrics["u_policy_net"],
                        top_frac,
                    ),
                    "target_oracle_top_hit_u": _top_hit(
                        valid_target["target_soft"],
                        valid_metrics["u_policy_net"],
                        top_frac,
                    ),
                    "model_top_mean_u": _top_mean(valid_score, valid_metrics["u_policy_net"], top_frac),
                    "model_top_hit_u": _top_hit(valid_score, valid_metrics["u_policy_net"], top_frac),
                }
                row.update(_quality_distribution(frame, spec.label_column, valid_bucket_mask))
                row.update(signal)
                row["model_vs_oracle_top_mean_u_gap"] = row["model_top_mean_u"] - row["target_oracle_top_mean_u"]
                row["diagnosis"] = _diagnose(row)
                summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    feature_ic = pd.concat(feature_ic_parts, ignore_index=True) if feature_ic_parts else pd.DataFrame()
    aggregate = _aggregate(summary, expected_months=len(months), min_bucket_rows=min_bucket_rows)

    paths = {
        "summary": output_dir / "source_tag_feature_learnability_summary.csv",
        "aggregate": output_dir / "source_tag_feature_learnability_aggregate.csv",
        "feature_ic": output_dir / "source_tag_feature_ic_by_bucket.csv",
        "manifest": output_dir / "manifest.json",
    }
    summary.to_csv(paths["summary"], index=False)
    aggregate.to_csv(paths["aggregate"], index=False)
    feature_ic.to_csv(paths["feature_ic"], index=False)

    manifest = {
        "scope": "source_tag_feature_learnability_diagnostic",
        "quality_labels_path": str(quality_labels_path),
        "labels_path": str(labels_path),
        "label_ablation_manifest": str(manifest_path),
        "output_dir": str(output_dir),
        "rows": int(len(frame)),
        "utility_source": metrics.attrs.get("utility_source"),
        "months": list(months),
        "ablations": [spec.name for spec in specs],
        "seeds": [int(seed) for seed in seeds],
        "top_frac": float(top_frac),
        "top_features": int(top_features),
        "min_bucket_rows": int(min_bucket_rows),
        "join_report": join_report,
        "feature_store": feature_report,
        "base_feature_count": int(len(base_features)),
        "source_feature_count": int(len(source_features)),
        "outputs": {key: str(path) for key, path in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    markdown = _write_report(output_dir, summary, aggregate, manifest)
    manifest["outputs"]["markdown"] = str(markdown)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-labels-path", type=Path, default=DEFAULT_QUALITY_LABELS)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--label-ablation-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--max-feature-store-features", type=int, default=96)
    parser.add_argument("--ablations", type=str, default=",".join(DEFAULT_ABLATIONS))
    parser.add_argument("--months", type=str, default=",".join(DEFAULT_MONTHS))
    parser.add_argument("--seeds", type=str, default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--train-lookback-months", type=int, default=None)
    parser.add_argument("--min-train-rows", type=int, default=500)
    parser.add_argument("--min-valid-rows", type=int, default=100)
    parser.add_argument("--min-bucket-rows", type=int, default=80)
    parser.add_argument("--top-frac", type=float, default=DEFAULT_TOP_FRAC)
    parser.add_argument("--top-features", type=int, default=DEFAULT_TOP_FEATURES)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_report(
        quality_labels_path=args.quality_labels_path,
        labels_path=args.labels_path,
        manifest_path=args.label_ablation_manifest,
        output_dir=args.output_dir,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        max_feature_store_features=args.max_feature_store_features,
        ablations=_parse_csv(args.ablations, DEFAULT_ABLATIONS),
        months=_parse_csv(args.months, DEFAULT_MONTHS),
        seeds=_parse_int_csv(args.seeds, DEFAULT_SEEDS),
        train_lookback_months=args.train_lookback_months,
        min_train_rows=int(args.min_train_rows),
        min_valid_rows=int(args.min_valid_rows),
        min_bucket_rows=int(args.min_bucket_rows),
        top_frac=float(args.top_frac),
        top_features=int(args.top_features),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
