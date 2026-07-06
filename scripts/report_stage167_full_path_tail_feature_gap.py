#!/usr/bin/env python3
"""Feature diagnostics for Stage167 full-path adverse tail.

This reconstructs the Stage167 causal two-head selector and compares selected
rows whose first-touch path is clean but whose full path later becomes dirty.
The goal is to decide whether current causal features can explain the
full-path/trailing-risk gap before changing labels again.
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

from scripts.diagnose_label_matched_clean_dirty_feature_gap import (  # noqa: E402
    _bucket_key,
    _eligible_masks,
    _feature_contrast,
    _feature_family,
    _table,
)
from scripts.run_first_touch_label_training_smoke import (  # noqa: E402
    _first_touch_eval_metrics,
    _target_from_frame,
)
from scripts.run_first_touch_two_head_training_smoke import (  # noqa: E402
    _final_score,
    _fit_seed_ensemble,
    _select_indices,
)
from scripts.run_label_quality_proxy_diagnostics import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    DEFAULT_FEATURE_LIST_CSV,
    _feature_columns,
    _json_safe,
    _load_feature_store_columns,
    _load_labels,
    _path_metrics,
    _read_feature_list,
    _safe_mean,
    _safe_quantile,
    _spearman,
)
from scripts.run_label_weighted_proxy_ablation import (  # noqa: E402
    _effective_sample_size,
    _weight_series,
)


DEFAULT_LABELS_PATH = Path("data_perp/artifacts/20260703_190000_clean_first_touch_tail_veto_stage167_labels/labels")
DEFAULT_OUTPUT_DIR = Path("data_perp/reports/stage167_full_path_tail_feature_gap_v1")
DEFAULT_UTILITY_TARGET_MODE = "column:__stage167_utility_target_soft__:__stage167_utility_target_hard__"
DEFAULT_SUPPORT_TARGET_MODE = "column:__stage167_risk_target_soft__:__stage167_risk_target_hard__"
DEFAULT_MONTHS = ("2026-04", "2026-05", "2026-06")


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


def _parse_int_csv(value: str | list[int] | tuple[int, ...]) -> list[int]:
    if isinstance(value, (list, tuple)):
        return [int(part) for part in value]
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def _sigmoid(values: Any, scale: float = 1.0) -> pd.Series:
    series = _safe_numeric(values).astype(float)
    arr = np.clip(series.to_numpy(dtype=np.float64) / float(scale), -60.0, 60.0)
    return pd.Series(1.0 / (1.0 + np.exp(-arr)), index=series.index)


def _full_path_target(frame: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    full_mae = _safe_numeric(frame.get("__first_touch_full_path_mae_to_sl__")).fillna(10.0).clip(lower=0.0)
    first_mae = _safe_numeric(metrics["first_touch_mae_to_sl"]).fillna(10.0).clip(lower=0.0)
    clean = _safe_numeric(metrics["clean_first_touch_exec"]).fillna(0.0).clip(0.0, 1.0)
    timeout = _safe_numeric(metrics["first_touch_timeout"].astype(float)).fillna(1.0).clip(0.0, 1.0)
    net = _safe_numeric(metrics["first_touch_net"]).fillna(-0.05)
    soft = (
        clean
        * (1.0 - timeout)
        * _sigmoid(0.75 - first_mae, scale=0.20)
        * _sigmoid(3.0 - full_mae, scale=0.80)
        * _sigmoid(net - 0.001, scale=0.004)
    ).clip(0.0, 1.0)
    hard = (clean >= 0.5) & (timeout < 0.5) & (net > 0.0) & (first_mae <= 1.0) & (full_mae <= 3.0)
    return pd.DataFrame(
        {"target_soft": soft.astype(np.float32), "target_hard": hard.astype(float).astype(np.float32)},
        index=frame.index,
    )


def _selected_group_masks(
    *,
    valid: pd.DataFrame,
    valid_metrics: pd.DataFrame,
    selected_mask: pd.Series,
    full_path_clean_r: float,
    full_path_dirty_r: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    full_mae = _safe_numeric(valid.get("__first_touch_full_path_mae_to_sl__")).fillna(10.0)
    first_mae = _safe_numeric(valid_metrics["first_touch_mae_to_sl"]).fillna(10.0)
    net = _safe_numeric(valid_metrics["first_touch_net"]).fillna(-0.05)
    clean_exec = _safe_numeric(valid_metrics["clean_first_touch_exec"]).fillna(0.0) >= 0.5
    timeout = _safe_numeric(valid_metrics["first_touch_timeout"].astype(float)).fillna(1.0) >= 0.5
    first_touch_clean = clean_exec & (~timeout) & (net > 0.0) & (first_mae <= 1.0)
    full_clean = selected_mask & first_touch_clean & (full_mae <= float(full_path_clean_r))
    full_dirty = selected_mask & (net > 0.0) & (full_mae >= float(full_path_dirty_r))
    return first_touch_clean.fillna(False), full_clean.fillna(False), full_dirty.fillna(False)


def _ledger_rows(
    *,
    valid: pd.DataFrame,
    valid_metrics: pd.DataFrame,
    utility_pred: pd.Series,
    support_pred: pd.Series,
    score: pd.Series,
    selected_idx: np.ndarray,
    month: str,
    utility_weight_arm: str,
    support_weight_arm: str,
    support_gate_frac: float,
    top_frac: float,
    score_rule: str,
) -> pd.DataFrame:
    if not len(selected_idx):
        return pd.DataFrame()
    out = valid.iloc[selected_idx][["__ts__", "__symbol__"]].reset_index(drop=True).copy()
    out["period"] = str(month)
    out["utility_weight_arm"] = str(utility_weight_arm)
    out["support_weight_arm"] = str(support_weight_arm)
    out["support_gate_frac"] = float(support_gate_frac)
    out["top_frac"] = float(top_frac)
    out["score_rule"] = str(score_rule)
    out["utility_pred"] = _safe_numeric(utility_pred).iloc[selected_idx].to_numpy(dtype=np.float64, copy=False)
    out["support_pred"] = _safe_numeric(support_pred).iloc[selected_idx].to_numpy(dtype=np.float64, copy=False)
    out["score"] = _safe_numeric(score).iloc[selected_idx].to_numpy(dtype=np.float64, copy=False)
    selected_metrics = valid_metrics.iloc[selected_idx].reset_index(drop=True)
    out["first_touch_net"] = _safe_numeric(selected_metrics["first_touch_net"]).to_numpy(dtype=np.float64, copy=False)
    out["clean_first_touch_exec"] = _safe_numeric(selected_metrics["clean_first_touch_exec"]).to_numpy(dtype=np.float64, copy=False)
    out["first_touch_timeout"] = _safe_numeric(selected_metrics["first_touch_timeout"].astype(float)).to_numpy(
        dtype=np.float64,
        copy=False,
    )
    out["first_touch_mae_to_sl"] = _safe_numeric(selected_metrics["first_touch_mae_to_sl"]).to_numpy(dtype=np.float64, copy=False)
    out["full_path_mae_to_sl"] = _safe_numeric(
        valid.iloc[selected_idx].reset_index(drop=True).get("__first_touch_full_path_mae_to_sl__"),
    ).to_numpy(dtype=np.float64, copy=False)
    out["full_path_mfe_to_tp"] = _safe_numeric(
        valid.iloc[selected_idx].reset_index(drop=True).get("__first_touch_full_path_mfe_to_tp__"),
    ).to_numpy(dtype=np.float64, copy=False)
    out["barrier"] = _safe_numeric(selected_metrics["barrier"]).to_numpy(dtype=np.float64, copy=False)
    out["side"] = _safe_numeric(selected_metrics["side"]).to_numpy(dtype=np.float64, copy=False)
    return out.sort_values(["period", "score"], ascending=[True, False])


def _mask_metrics(
    *,
    valid: pd.DataFrame,
    metrics: pd.DataFrame,
    mask: pd.Series,
) -> dict[str, Any]:
    subset = metrics.loc[mask]
    frame_subset = valid.loc[mask]
    full_mae = _safe_numeric(frame_subset.get("__first_touch_full_path_mae_to_sl__"))
    return {
        "rows": int(mask.sum()),
        "mean_first_touch_net": _safe_mean(subset.get("first_touch_net")),
        "hit_first_touch_net": _safe_mean(_safe_numeric(subset.get("first_touch_net")) > 0.0),
        "clean_first_touch_exec_rate": _safe_mean(subset.get("clean_first_touch_exec")),
        "first_touch_timeout_rate": _safe_mean(_safe_numeric(subset.get("first_touch_timeout").astype(float)) if len(subset) else []),
        "bad_first_touch_mae_to_sl_rate": _safe_mean(_safe_numeric(subset.get("first_touch_mae_to_sl")) >= 1.0),
        "p90_first_touch_mae_to_sl": _safe_quantile(subset.get("first_touch_mae_to_sl"), 0.90),
        "bad_full_path_mae_3r_rate": _safe_mean(full_mae >= 3.0),
        "p90_full_path_mae_to_sl": _safe_quantile(full_mae, 0.90),
    }


def _family_summary(contrast: pd.DataFrame) -> pd.DataFrame:
    if contrast.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for family, group in contrast.groupby("feature_family", dropna=False, sort=False):
        top_features = (
            group.sort_values(["best_auc", "abs_bucket_gap"], ascending=[False, False])["feature"]
            .drop_duplicates()
            .head(8)
            .astype(str)
            .tolist()
        )
        rows.append(
            {
                "feature_family": family,
                "rows": int(len(group)),
                "months": ",".join(sorted(group["month"].astype(str).unique().tolist())),
                "mean_best_auc": _safe_mean(group["best_auc"]),
                "max_best_auc": _safe_quantile(group["best_auc"], 1.0),
                "mean_abs_bucket_gap": _safe_mean(group["abs_bucket_gap"]),
                "top_features": ",".join(top_features),
            }
        )
    return pd.DataFrame(rows).sort_values(["max_best_auc", "rows"], ascending=[False, False])


def _write_markdown(
    *,
    output_dir: Path,
    summary: pd.DataFrame,
    contrast: pd.DataFrame,
    family: pd.DataFrame,
    manifest: dict[str, Any],
) -> Path:
    path = output_dir / "stage167_full_path_tail_feature_gap.md"
    summary_cols = [
        "period",
        "selected_rows",
        "first_touch_clean_rows",
        "full_path_clean_rows",
        "full_path_dirty_rows",
        "full_path_dirty_rate",
        "selected_mean_first_touch_net",
        "selected_clean_first_touch_exec_rate",
        "selected_p90_first_touch_mae_to_sl",
        "selected_p90_full_path_mae_to_sl",
        "matched_full_path_clean_rows",
        "matched_full_path_dirty_rows",
        "matched_bucket_count",
        "top_feature",
        "top_feature_family",
        "top_feature_best_auc",
        "top_feature_direction",
    ]
    contrast_cols = [
        "month",
        "match_mode",
        "feature",
        "feature_family",
        "best_auc",
        "best_direction",
        "bucket_equal_weight_gap_mean",
        "bucket_gap_positive_rate",
        "train_label_ic",
        "valid_label_ic",
        "valid_utility_ic",
        "valid_score_ic",
    ]
    family_cols = [
        "feature_family",
        "rows",
        "months",
        "mean_best_auc",
        "max_best_auc",
        "mean_abs_bucket_gap",
        "top_features",
    ]
    lines = [
        "# Stage167 Full-Path Tail Feature Gap",
        "",
        "Scope: causal diagnostic. The Stage167 two-head selector is reconstructed month-forward, then selected full-path-clean rows are contrasted with selected full-path-dirty rows. No production training, Optuna, or policy optimisation is run.",
        "",
        f"Labels: `{manifest['labels_path']}`",
        f"Feature dir: `{manifest['feature_dir']}`",
        f"Months: `{', '.join(manifest['months'])}`",
        f"Selector: utility `{manifest['utility_weight_arm']}`, support `{manifest['support_weight_arm']}`, gate `{manifest['support_gate_frac']}`, top `{manifest['top_frac']}`",
        f"Full-path clean <= `{manifest['full_path_clean_r']}R`; dirty >= `{manifest['full_path_dirty_r']}R`",
        "",
        "## Monthly Selected Groups",
        "",
        _table(summary, summary_cols, limit=80),
        "",
        "## Feature Families",
        "",
        _table(family, family_cols, limit=40),
        "",
        "## Strongest Feature Separators",
        "",
        _table(contrast.sort_values(["best_auc", "abs_bucket_gap"], ascending=[False, False]) if not contrast.empty else contrast, contrast_cols, limit=100),
        "",
        "## Outputs",
        "",
        f"- Summary: `{manifest['outputs']['summary']}`",
        f"- Feature contrast: `{manifest['outputs']['feature_contrast']}`",
        f"- Family summary: `{manifest['outputs']['family_summary']}`",
        f"- Selected ledger: `{manifest['outputs']['selected_ledger']}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_report(
    *,
    labels_path: Path,
    output_dir: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    max_feature_store_features: int | None,
    months: list[str],
    utility_target_mode: str,
    support_target_mode: str,
    utility_weight_arm: str,
    support_weight_arm: str,
    support_gate_frac: float,
    top_frac: float,
    score_rule: str,
    seeds: list[int],
    match_mode: str,
    min_class_rows: int,
    full_path_clean_r: float,
    full_path_dirty_r: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _load_labels(labels_path)
    selected_features = _read_feature_list(feature_list_csv, max_features=max_feature_store_features)
    feature_matrix, feature_store_report = _load_feature_store_columns(
        frame,
        feature_dir=feature_dir,
        selected_features=selected_features,
    )
    if not feature_matrix.empty:
        new_cols = [col for col in feature_matrix.columns if col not in frame.columns]
        if new_cols:
            frame = pd.concat(
                [
                    frame.reset_index(drop=True),
                    feature_matrix.loc[:, new_cols].reset_index(drop=True).astype(np.float32, copy=False),
                ],
                axis=1,
            )
    metrics = _first_touch_eval_metrics(frame, _path_metrics(frame))
    utility_target = _target_from_frame(frame, metrics, target_mode=utility_target_mode)
    support_target = _target_from_frame(frame, metrics, target_mode=support_target_mode)
    full_path_target = _full_path_target(frame, metrics)
    features = _feature_columns(frame)

    month_ser = frame["__ts__"].dt.to_period("M").astype(str)
    summary_rows: list[dict[str, Any]] = []
    contrast_parts: list[pd.DataFrame] = []
    ledger_parts: list[pd.DataFrame] = []
    diag_rows: list[dict[str, Any]] = []

    x = frame[features].replace([np.inf, -np.inf], np.nan)
    for month in months:
        train_mask = month_ser < str(month)
        valid_mask = month_ser == str(month)
        if int(train_mask.sum()) < 500 or int(valid_mask.sum()) < 100:
            summary_rows.append(
                {
                    "period": str(month),
                    "skipped": True,
                    "train_rows": int(train_mask.sum()),
                    "valid_rows": int(valid_mask.sum()),
                }
            )
            continue
        med = x.loc[train_mask].median(numeric_only=True)
        x_filled = x.fillna(med).fillna(0.0).astype(np.float32, copy=False)
        train = frame.loc[train_mask].copy()
        train_metrics = metrics.loc[train_mask].copy()
        valid = frame.loc[valid_mask].copy().reset_index(drop=True)
        valid_metrics = metrics.loc[valid_mask].copy().reset_index(drop=True)
        valid_full_path_target = full_path_target.loc[valid_mask].copy().reset_index(drop=True)

        utility_train_target = utility_target.loc[train_mask].copy()
        utility_weights = _weight_series(
            frame=train,
            metrics=train_metrics,
            target=utility_train_target,
            arm=utility_weight_arm,
        )
        utility_pred, utility_seed_std = _fit_seed_ensemble(
            x_train=x_filled.loc[train_mask],
            y_train=utility_train_target["target_soft"],
            w_train=utility_weights,
            x_valid=x_filled.loc[valid_mask],
            seeds=seeds,
        )

        support_train_target = support_target.loc[train_mask].copy()
        support_weights = _weight_series(
            frame=train,
            metrics=train_metrics,
            target=support_train_target,
            arm=support_weight_arm,
        )
        support_pred, support_seed_std = _fit_seed_ensemble(
            x_train=x_filled.loc[train_mask],
            y_train=support_train_target["target_soft"],
            w_train=support_weights,
            x_valid=x_filled.loc[valid_mask],
            seeds=seeds,
        )
        score = _final_score(utility_pred=utility_pred, support_pred=support_pred, score_rule=score_rule)
        selected_idx, _gate_mask = _select_indices(
            score=score,
            support_pred=support_pred,
            support_gate_frac=float(support_gate_frac),
            top_frac=float(top_frac),
            total_rows=len(valid),
        )
        selected_mask = pd.Series(False, index=valid.index)
        if len(selected_idx):
            selected_mask.iloc[selected_idx] = True
        first_touch_clean, full_clean, full_dirty = _selected_group_masks(
            valid=valid,
            valid_metrics=valid_metrics,
            selected_mask=selected_mask,
            full_path_clean_r=full_path_clean_r,
            full_path_dirty_r=full_path_dirty_r,
        )
        selected_metrics = _mask_metrics(valid=valid, metrics=valid_metrics, mask=selected_mask)
        clean_metrics = _mask_metrics(valid=valid, metrics=valid_metrics, mask=full_clean)
        dirty_metrics = _mask_metrics(valid=valid, metrics=valid_metrics, mask=full_dirty)
        bucket = _bucket_key(valid, valid_metrics, match_mode)
        matched_clean, matched_dirty, eligible, eligible_bucket_count = _eligible_masks(
            bucket=bucket,
            missed_clean_mask=full_clean,
            dirty_false_positive_mask=full_dirty,
        )
        contrast = _feature_contrast(
            train=train,
            valid=valid,
            valid_metrics=valid_metrics,
            features=features,
            target_train=full_path_target.loc[train_mask, "target_soft"],
            target_valid=valid_full_path_target["target_soft"],
            score=score,
            bucket=bucket,
            clean_mask=matched_clean,
            dirty_mask=matched_dirty,
            eligible_candidate_mask=eligible,
            proxy_features=[],
            min_class_rows=int(min_class_rows),
            source="stage167_selected",
            month=str(month),
            label_arm="stage167_full_path_tail",
            top_frac=float(top_frac),
            match_mode=str(match_mode),
        )
        if not contrast.empty:
            contrast["selector"] = "stage167_two_head_causal_smoke"
            contrast_parts.append(contrast)

        row: dict[str, Any] = {
            "period": str(month),
            "train_rows": int(train_mask.sum()),
            "valid_rows": int(valid_mask.sum()),
            "selected_rows": int(selected_mask.sum()),
            "first_touch_clean_rows": int((selected_mask & first_touch_clean).sum()),
            "full_path_clean_rows": int(full_clean.sum()),
            "full_path_dirty_rows": int(full_dirty.sum()),
            "full_path_dirty_rate": float(full_dirty.sum() / selected_mask.sum()) if int(selected_mask.sum()) else 0.0,
            "matched_full_path_clean_rows": int(matched_clean.sum()),
            "matched_full_path_dirty_rows": int(matched_dirty.sum()),
            "matched_bucket_count": int(eligible_bucket_count),
            "selected_mean_first_touch_net": selected_metrics["mean_first_touch_net"],
            "selected_clean_first_touch_exec_rate": selected_metrics["clean_first_touch_exec_rate"],
            "selected_first_touch_timeout_rate": selected_metrics["first_touch_timeout_rate"],
            "selected_bad_first_touch_mae_to_sl_rate": selected_metrics["bad_first_touch_mae_to_sl_rate"],
            "selected_p90_first_touch_mae_to_sl": selected_metrics["p90_first_touch_mae_to_sl"],
            "selected_bad_full_path_mae_3r_rate": selected_metrics["bad_full_path_mae_3r_rate"],
            "selected_p90_full_path_mae_to_sl": selected_metrics["p90_full_path_mae_to_sl"],
            "full_path_clean_mean_first_touch_net": clean_metrics["mean_first_touch_net"],
            "full_path_dirty_mean_first_touch_net": dirty_metrics["mean_first_touch_net"],
            "score_ic_full_path_target": _spearman(score, valid_full_path_target["target_soft"]),
            "score_ic_full_path_dirty": _spearman(
                score,
                _safe_numeric(valid.get("__first_touch_full_path_mae_to_sl__")).ge(float(full_path_dirty_r)).astype(float),
            ),
            "utility_seed_std": utility_seed_std,
            "support_seed_std": support_seed_std,
            "utility_weight_effective_frac": _effective_sample_size(utility_weights) / float(len(utility_weights)),
            "support_weight_effective_frac": _effective_sample_size(support_weights) / float(len(support_weights)),
        }
        if not contrast.empty:
            top = contrast.iloc[0]
            row.update(
                {
                    "top_feature": top.get("feature"),
                    "top_feature_family": top.get("feature_family"),
                    "top_feature_best_auc": top.get("best_auc"),
                    "top_feature_direction": top.get("best_direction"),
                }
            )
        summary_rows.append(row)
        ledger_parts.append(
            _ledger_rows(
                valid=valid,
                valid_metrics=valid_metrics,
                utility_pred=utility_pred,
                support_pred=support_pred,
                score=score,
                selected_idx=selected_idx,
                month=str(month),
                utility_weight_arm=utility_weight_arm,
                support_weight_arm=support_weight_arm,
                support_gate_frac=support_gate_frac,
                top_frac=top_frac,
                score_rule=score_rule,
            )
        )
        diag_rows.extend(
            [
                {
                    "period": str(month),
                    "head": "utility",
                    "target_mode": utility_target_mode,
                    "weight_arm": utility_weight_arm,
                    "seed_std_mean": utility_seed_std,
                    "weight_effective_frac": _effective_sample_size(utility_weights) / float(len(utility_weights)),
                },
                {
                    "period": str(month),
                    "head": "support",
                    "target_mode": support_target_mode,
                    "weight_arm": support_weight_arm,
                    "seed_std_mean": support_seed_std,
                    "weight_effective_frac": _effective_sample_size(support_weights) / float(len(support_weights)),
                },
            ]
        )

    summary = pd.DataFrame(summary_rows)
    contrast = pd.concat(contrast_parts, ignore_index=True) if contrast_parts else pd.DataFrame()
    family = _family_summary(contrast)
    ledger = pd.concat(ledger_parts, ignore_index=True) if ledger_parts else pd.DataFrame()
    diagnostics = pd.DataFrame(diag_rows)

    paths = {
        "summary": output_dir / "stage167_full_path_tail_feature_gap_summary.csv",
        "feature_contrast": output_dir / "stage167_full_path_tail_feature_contrast.csv",
        "family_summary": output_dir / "stage167_full_path_tail_feature_family_summary.csv",
        "selected_ledger": output_dir / "stage167_full_path_tail_selected_ledger.csv",
        "diagnostics": output_dir / "stage167_full_path_tail_training_diagnostics.csv",
        "manifest": output_dir / "manifest.json",
    }
    summary.to_csv(paths["summary"], index=False)
    contrast.to_csv(paths["feature_contrast"], index=False)
    family.to_csv(paths["family_summary"], index=False)
    ledger.to_csv(paths["selected_ledger"], index=False)
    diagnostics.to_csv(paths["diagnostics"], index=False)
    manifest = {
        "scope": "stage167_full_path_tail_feature_gap",
        "labels_path": str(labels_path),
        "output_dir": str(output_dir),
        "feature_dir": str(feature_dir),
        "feature_list_csv": str(feature_list_csv),
        "max_feature_store_features": max_feature_store_features,
        "feature_count": int(len(features)),
        "feature_store": feature_store_report,
        "months": list(months),
        "utility_target_mode": str(utility_target_mode),
        "support_target_mode": str(support_target_mode),
        "utility_weight_arm": str(utility_weight_arm),
        "support_weight_arm": str(support_weight_arm),
        "support_gate_frac": float(support_gate_frac),
        "top_frac": float(top_frac),
        "score_rule": str(score_rule),
        "seeds": list(seeds),
        "match_mode": str(match_mode),
        "min_class_rows": int(min_class_rows),
        "full_path_clean_r": float(full_path_clean_r),
        "full_path_dirty_r": float(full_path_dirty_r),
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    markdown = _write_markdown(output_dir=output_dir, summary=summary, contrast=contrast, family=family, manifest=manifest)
    manifest["outputs"]["markdown"] = str(markdown)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--max-feature-store-features", type=int, default=160)
    parser.add_argument("--months", default=",".join(DEFAULT_MONTHS))
    parser.add_argument("--utility-target-mode", default=DEFAULT_UTILITY_TARGET_MODE)
    parser.add_argument("--support-target-mode", default=DEFAULT_SUPPORT_TARGET_MODE)
    parser.add_argument("--utility-weight-arm", default="W0_base")
    parser.add_argument("--support-weight-arm", default="W7_timestamp_balanced")
    parser.add_argument("--support-gate-frac", type=float, default=0.02)
    parser.add_argument("--top-frac", type=float, default=0.005)
    parser.add_argument("--score-rule", default="utility_inside_support")
    parser.add_argument("--seeds", default="42,7301,999")
    parser.add_argument("--match-mode", default="day_side")
    parser.add_argument("--min-class-rows", type=int, default=3)
    parser.add_argument("--full-path-clean-r", type=float, default=3.0)
    parser.add_argument("--full-path-dirty-r", type=float, default=3.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_report(
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        max_feature_store_features=args.max_feature_store_features,
        months=_parse_csv(str(args.months), default=DEFAULT_MONTHS),
        utility_target_mode=str(args.utility_target_mode),
        support_target_mode=str(args.support_target_mode),
        utility_weight_arm=str(args.utility_weight_arm),
        support_weight_arm=str(args.support_weight_arm),
        support_gate_frac=float(args.support_gate_frac),
        top_frac=float(args.top_frac),
        score_rule=str(args.score_rule),
        seeds=_parse_int_csv(str(args.seeds)),
        match_mode=str(args.match_mode),
        min_class_rows=int(args.min_class_rows),
        full_path_clean_r=float(args.full_path_clean_r),
        full_path_dirty_r=float(args.full_path_dirty_r),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
