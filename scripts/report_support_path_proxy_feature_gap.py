#!/usr/bin/env python3
"""Feature-gap diagnostic for support/path-decisive label proxies.

This is a proxy-only report. It reconstructs the same causal target and proxy
scores used by ``run_support_path_decisive_label_ablation.py``, then compares
oracle target rows missed by the proxy against dirty proxy-selected false
positives. The intent is to diagnose feature observability before training
base/meta models.
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
    DEFAULT_LABELS_PATH,
    _bucket_key,
    _build_frame,
    _eligible_masks,
    _feature_contrast,
    _feature_family,
    _mfe_mae,
    _table,
)
from scripts.report_label_proxy_quality_with_economic_limits import (  # noqa: E402
    DEFAULT_FEATURE_LIST_CSV,
    DEFAULT_PROXY_TOP_K,
    _score_proxy,
    _top_gate,
)
from scripts.run_label_quality_proxy_diagnostics import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    _feature_columns,
    _json_safe,
    _safe_mean,
    _safe_quantile,
    _selection_metrics,
    _spearman,
)
from scripts.run_soft_label_candidate_source_ablation import (  # noqa: E402
    _causal_time_edge_prior_features,
    _source_context,
)
from scripts.run_soft_label_economic_proxy_ablation import (  # noqa: E402
    DEFAULT_EVENT_FEATURE_STORE_FEATURES,
    DEFAULT_PRIOR_WINDOWS_DAYS,
    DEFAULT_STATE_PATH_PRIOR_FEATURES,
)
from scripts.run_support_path_decisive_label_ablation import (  # noqa: E402
    SupportArm,
    _default_arms,
    _target_for_arm,
)


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/support_path_proxy_feature_gap_stage141_v1")
DEFAULT_MONTHS = ("2026-06",)
DEFAULT_ARMS = (
    "P1_clean_support_14d",
    "P4_bounded_rebound_support_30d",
    "P6_decisive_time_edge_support_30d",
    "P7_clean_support_gate_14d",
)
DEFAULT_TOP_FRACS = (0.02, 0.03, 0.05)
DEFAULT_SELECTORS = ("fit_ic_proxy_oos", "fit_ic_proxy_inverse_oos")
DEFAULT_MATCH_MODES = ("day_side", "regime_side")


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


def _safe_numeric(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _selected_metrics(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    target: pd.DataFrame,
    score: pd.Series,
    selector: str,
    label_arm: str,
    month: str,
    top_frac: float,
) -> dict[str, Any]:
    out = _selection_metrics(
        frame=frame,
        metrics=metrics,
        target=target,
        score=score,
        arm=f"{selector}::{label_arm}",
        selector=selector,
        period=month,
        top_frac=float(top_frac),
    )
    return {
        "mean_return_net": out.get("mean_return_net"),
        "bad_mae_1r_rate": out.get("bad_mae_1r_rate"),
        "p90_mae_norm": out.get("p90_mae_norm"),
        "timeout_rate": out.get("timeout_rate"),
        "wide_barrier_25bps_rate": out.get("wide_barrier_25bps_rate"),
        "selected_rows": out.get("selected_rows"),
    }


def _mask_metrics(metrics: pd.DataFrame, mask: pd.Series) -> dict[str, Any]:
    mask = mask.fillna(False).astype(bool)
    selected = metrics.loc[mask]
    if selected.empty:
        return {
            "rows": 0,
            "mean_u": float("nan"),
            "bad_mae_1r_rate": float("nan"),
            "p90_mae_norm": float("nan"),
            "timeout_rate": float("nan"),
            "wide_25bps_rate": float("nan"),
        }
    return {
        "rows": int(mask.sum()),
        "mean_u": _safe_mean(selected["u_policy_net"]),
        "bad_mae_1r_rate": _safe_mean(selected["mae_norm"].ge(1.0).astype(float)),
        "p90_mae_norm": _safe_quantile(selected["mae_norm"], 0.90),
        "timeout_rate": _safe_mean(selected["is_timeout"].astype(float)),
        "wide_25bps_rate": _safe_mean(selected["barrier"].gt(0.025).astype(float)),
    }


def _dirty_execution(metrics: pd.DataFrame) -> pd.Series:
    mfe_mae = _mfe_mae(metrics)
    return (
        _safe_numeric(metrics["u_policy_net"]).le(0.0)
        | _safe_numeric(metrics["mae_norm"]).ge(1.0)
        | _safe_numeric(metrics["barrier"]).gt(0.025)
        | metrics["is_timeout"].astype(bool)
        | mfe_mae.lt(1.25)
    ).fillna(True)


def _summary_row(
    *,
    valid: pd.DataFrame,
    valid_metrics: pd.DataFrame,
    target_valid: pd.DataFrame,
    score: pd.Series,
    selector_mask: pd.Series,
    oracle_mask: pd.Series,
    missed_oracle: pd.Series,
    false_dirty: pd.Series,
    matched_oracle: pd.Series,
    matched_dirty: pd.Series,
    eligible_buckets: int,
    month: str,
    label_arm: str,
    selector: str,
    top_frac: float,
    match_mode: str,
    proxy_features: list[str],
    contrast: pd.DataFrame,
) -> dict[str, Any]:
    selector_metrics = _selected_metrics(
        frame=valid,
        metrics=valid_metrics,
        target=target_valid,
        score=score,
        selector=selector,
        label_arm=label_arm,
        month=month,
        top_frac=top_frac,
    )
    oracle_metrics = _selected_metrics(
        frame=valid,
        metrics=valid_metrics,
        target=target_valid,
        score=target_valid["target_soft"],
        selector="oracle_target_sort",
        label_arm=label_arm,
        month=month,
        top_frac=top_frac,
    )
    recovered = oracle_mask & selector_mask
    missed_metrics = _mask_metrics(valid_metrics, missed_oracle)
    false_metrics = _mask_metrics(valid_metrics, false_dirty)
    row: dict[str, Any] = {
        "month": month,
        "label_arm": label_arm,
        "selector": selector,
        "top_frac": float(top_frac),
        "match_mode": match_mode,
        "valid_rows": int(len(valid)),
        "oracle_top_rows": int(oracle_mask.sum()),
        "selector_top_rows": int(selector_mask.sum()),
        "recovered_oracle_rows": int(recovered.sum()),
        "oracle_recovery_rate": float(recovered.sum() / oracle_mask.sum()) if int(oracle_mask.sum()) else 0.0,
        "missed_oracle_rows": int(missed_oracle.sum()),
        "false_dirty_rows": int(false_dirty.sum()),
        "matched_missed_oracle_rows": int(matched_oracle.sum()),
        "matched_false_dirty_rows": int(matched_dirty.sum()),
        "matched_bucket_count": int(eligible_buckets),
        "score_ic_target": _spearman(score, target_valid["target_soft"]),
        "score_ic_u": _spearman(score, valid_metrics["u_policy_net"]),
        "score_ic_bad_mae": _spearman(score, valid_metrics["mae_norm"].ge(1.0).astype(float)),
        "oracle_mean_return_net": oracle_metrics.get("mean_return_net"),
        "oracle_bad_mae_1r_rate": oracle_metrics.get("bad_mae_1r_rate"),
        "oracle_p90_mae_norm": oracle_metrics.get("p90_mae_norm"),
        "oracle_timeout_rate": oracle_metrics.get("timeout_rate"),
        "selector_mean_return_net": selector_metrics.get("mean_return_net"),
        "selector_bad_mae_1r_rate": selector_metrics.get("bad_mae_1r_rate"),
        "selector_p90_mae_norm": selector_metrics.get("p90_mae_norm"),
        "selector_timeout_rate": selector_metrics.get("timeout_rate"),
        "missed_oracle_mean_u": missed_metrics["mean_u"],
        "missed_oracle_bad_mae_1r_rate": missed_metrics["bad_mae_1r_rate"],
        "false_dirty_mean_u": false_metrics["mean_u"],
        "false_dirty_bad_mae_1r_rate": false_metrics["bad_mae_1r_rate"],
        "proxy_features": ",".join(proxy_features),
    }
    if not contrast.empty:
        top = contrast.iloc[0]
        row.update(
            {
                "top_feature": top.get("feature"),
                "top_feature_family": top.get("feature_family"),
                "top_feature_best_auc": top.get("best_auc"),
                "top_feature_direction": top.get("best_direction"),
                "top_feature_bucket_gap": top.get("bucket_equal_weight_gap_mean"),
                "top_feature_is_proxy_feature": bool(top.get("is_proxy_feature")),
            }
        )
    return row


def _family_summary(contrast: pd.DataFrame, min_best_auc: float) -> pd.DataFrame:
    if contrast.empty:
        return pd.DataFrame()
    strong = contrast[contrast["best_auc"].ge(float(min_best_auc))].copy()
    if strong.empty:
        strong = contrast.sort_values("best_auc", ascending=False).head(100).copy()
    rows: list[dict[str, Any]] = []
    for family, group in strong.groupby("feature_family", dropna=False, sort=False):
        rows.append(
            {
                "feature_family": str(family),
                "rows": int(len(group)),
                "selectors": ",".join(sorted(group["source"].astype(str).unique())),
                "months": ",".join(sorted(group["month"].astype(str).unique())),
                "label_arms": ",".join(sorted(group["label_arm"].astype(str).unique())),
                "mean_best_auc": _safe_mean(group["best_auc"]),
                "max_best_auc": _safe_quantile(group["best_auc"], 1.0),
                "mean_abs_bucket_gap": _safe_mean(group["abs_bucket_gap"]),
                "proxy_feature_share": _safe_mean(group["is_proxy_feature"].astype(float)),
                "top_features": ",".join(
                    group.sort_values("best_auc", ascending=False)["feature"]
                    .drop_duplicates()
                    .head(8)
                    .astype(str)
                    .tolist()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["max_best_auc", "rows"], ascending=[False, False])


def _write_markdown(
    *,
    output_dir: Path,
    summary: pd.DataFrame,
    contrast: pd.DataFrame,
    family_summary: pd.DataFrame,
    manifest: dict[str, Any],
) -> Path:
    path = output_dir / "support_path_proxy_feature_gap.md"
    best_summary = (
        summary.sort_values(
            ["oracle_recovery_rate", "selector_mean_return_net"],
            ascending=[True, False],
        )
        if not summary.empty
        else summary
    )
    lines = [
        "# Support Path Proxy Feature Gap",
        "",
        "Scope: proxy-only feature-gap diagnostic. No base/meta training, Optuna, or policy optimisation.",
        "",
        f"Labels: `{manifest['labels_path']}`",
        f"Months: `{', '.join(manifest['months'])}`. Arms: `{', '.join(manifest['arms'])}`.",
        f"Selectors: `{', '.join(manifest['selectors'])}`. Top fractions: `{manifest['top_fracs']}`.",
        "",
        "## Summary",
        "",
        _table(
            best_summary,
            [
                "month",
                "label_arm",
                "selector",
                "top_frac",
                "match_mode",
                "oracle_recovery_rate",
                "missed_oracle_rows",
                "false_dirty_rows",
                "matched_missed_oracle_rows",
                "matched_false_dirty_rows",
                "score_ic_target",
                "score_ic_u",
                "score_ic_bad_mae",
                "oracle_mean_return_net",
                "selector_mean_return_net",
                "selector_bad_mae_1r_rate",
                "top_feature",
                "top_feature_family",
                "top_feature_best_auc",
                "top_feature_direction",
            ],
            limit=120,
        ),
        "",
        "## Strongest Separators",
        "",
        _table(
            contrast.sort_values(["best_auc", "abs_bucket_gap"], ascending=[False, False])
            if not contrast.empty
            else contrast,
            [
                "source",
                "month",
                "label_arm",
                "top_frac",
                "match_mode",
                "feature",
                "feature_family",
                "is_proxy_feature",
                "clean_rows",
                "dirty_rows",
                "best_auc",
                "best_direction",
                "bucket_equal_weight_gap_mean",
                "clean_median",
                "dirty_median",
                "valid_utility_ic",
                "valid_score_ic",
            ],
            limit=100,
        ),
        "",
        "## Feature Families",
        "",
        _table(
            family_summary,
            [
                "feature_family",
                "rows",
                "selectors",
                "label_arms",
                "mean_best_auc",
                "max_best_auc",
                "mean_abs_bucket_gap",
                "proxy_feature_share",
                "top_features",
            ],
            limit=80,
        ),
        "",
        "## Outputs",
        "",
        f"- Summary: `{manifest['outputs']['summary']}`",
        f"- Feature contrast: `{manifest['outputs']['feature_contrast']}`",
        f"- Family summary: `{manifest['outputs']['family_summary']}`",
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
    max_feature_columns: int | None,
    months: list[str],
    arms: list[str],
    top_fracs: list[float],
    selectors: list[str],
    match_modes: list[str],
    proxy_top_k: int,
    min_train_rows: int,
    min_valid_rows: int,
    min_class_rows: int,
    strong_auc_threshold: float,
    include_causal_time_edge_priors: bool,
    include_causal_outcome_priors: bool,
    include_causal_state_path_priors: bool,
    include_event_confirmation_features: bool,
    include_adverse_path_composites: bool,
    prior_windows_days: list[float],
    prior_embargo_hours: float,
    state_path_prior_features: list[str],
    event_feature_store_features: list[str],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame, metrics, reports = _build_frame(
        labels_path=labels_path,
        feature_dir=feature_dir,
        feature_list_csv=feature_list_csv,
        max_feature_store_features=max_feature_store_features,
        include_causal_outcome_priors=include_causal_outcome_priors,
        include_causal_state_path_priors=include_causal_state_path_priors,
        include_event_confirmation_features=include_event_confirmation_features,
        include_adverse_path_composites=include_adverse_path_composites,
        prior_windows_days=prior_windows_days,
        prior_embargo_hours=prior_embargo_hours,
        state_path_prior_features=state_path_prior_features,
        event_feature_store_features=event_feature_store_features,
    )
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    reports["causal_time_edge_priors"] = {"enabled": False}
    if include_causal_time_edge_priors:
        time_edge, reports["causal_time_edge_priors"] = _causal_time_edge_prior_features(
            frame,
            metrics,
            windows_days=prior_windows_days,
            embargo_hours=prior_embargo_hours,
        )
        frame = pd.concat([frame, time_edge.astype(np.float32, copy=False)], axis=1).copy()
    context = _source_context(frame)
    frame = pd.concat([frame, context.astype(np.float32, copy=False)], axis=1).copy()

    features = _feature_columns(frame)
    if max_feature_columns is not None and int(max_feature_columns) > 0:
        features = features[: int(max_feature_columns)]

    all_arms = {arm.name: arm for arm in _default_arms()}
    unknown = sorted(set(arms).difference(all_arms))
    if unknown:
        raise ValueError(f"Unknown arms: {unknown}. Available: {sorted(all_arms)}")
    selected_arms: list[SupportArm] = [all_arms[name] for name in arms]

    period = frame["__ts__"].dt.to_period("M").astype(str)
    summary_rows: list[dict[str, Any]] = []
    contrast_frames: list[pd.DataFrame] = []

    for month in months:
        train_mask = period.lt(str(month))
        valid_mask = period.eq(str(month))
        if int(train_mask.sum()) < int(min_train_rows) or int(valid_mask.sum()) < int(min_valid_rows):
            continue
        train = frame.loc[train_mask].copy()
        valid_raw = frame.loc[valid_mask].copy()
        valid = valid_raw.reset_index(drop=True)
        valid_metrics = metrics.loc[valid_mask].copy().reset_index(drop=True)
        dirty = _dirty_execution(valid_metrics)

        for arm in selected_arms:
            target = _target_for_arm(frame, metrics, arm)
            target.index = frame.index
            target_train = target.loc[train_mask, "target_soft"]
            target_valid = target.loc[valid_mask].copy().reset_index(drop=True)
            proxy_score, diag = _score_proxy(
                train=train,
                valid=valid_raw,
                features=features,
                y_train=target_train,
                proxy_top_k=int(proxy_top_k),
            )
            proxy_score = proxy_score.reset_index(drop=True)
            proxy_features = [str(v) for v in diag.get("proxy_features", [])]
            selector_scores = {
                "fit_ic_proxy_oos": proxy_score,
                "fit_ic_proxy_inverse_oos": -proxy_score,
            }

            for selector in selectors:
                if selector not in selector_scores:
                    raise ValueError(f"Unknown selector: {selector}")
                score = _safe_numeric(selector_scores[selector]).reset_index(drop=True)
                for top_frac in top_fracs:
                    oracle_mask = _top_gate(target_valid["target_soft"], float(top_frac)).reset_index(drop=True)
                    selector_mask = _top_gate(score, float(top_frac)).reset_index(drop=True)
                    target_hard = _safe_numeric(target_valid["target_hard"]).gt(0.5)
                    missed_oracle = oracle_mask & ~selector_mask & target_hard
                    false_dirty = selector_mask & ~oracle_mask & dirty
                    for match_mode in match_modes:
                        bucket = _bucket_key(valid, valid_metrics, match_mode).reset_index(drop=True)
                        matched_oracle, matched_dirty, eligible, eligible_buckets = _eligible_masks(
                            bucket=bucket,
                            missed_clean_mask=missed_oracle,
                            dirty_false_positive_mask=false_dirty,
                        )
                        contrast = _feature_contrast(
                            train=train,
                            valid=valid,
                            valid_metrics=valid_metrics,
                            features=features,
                            target_train=target_train,
                            target_valid=target_valid["target_soft"],
                            score=score,
                            bucket=bucket,
                            clean_mask=matched_oracle,
                            dirty_mask=matched_dirty,
                            eligible_candidate_mask=eligible,
                            proxy_features=proxy_features,
                            min_class_rows=int(min_class_rows),
                            source=selector,
                            month=str(month),
                            label_arm=arm.name,
                            top_frac=float(top_frac),
                            match_mode=match_mode,
                        )
                        if not contrast.empty:
                            contrast_frames.append(contrast)
                        summary_rows.append(
                            _summary_row(
                                valid=valid,
                                valid_metrics=valid_metrics,
                                target_valid=target_valid,
                                score=score,
                                selector_mask=selector_mask,
                                oracle_mask=oracle_mask,
                                missed_oracle=missed_oracle,
                                false_dirty=false_dirty,
                                matched_oracle=matched_oracle,
                                matched_dirty=matched_dirty,
                                eligible_buckets=eligible_buckets,
                                month=str(month),
                                label_arm=arm.name,
                                selector=selector,
                                top_frac=float(top_frac),
                                match_mode=match_mode,
                                proxy_features=proxy_features,
                                contrast=contrast,
                            )
                        )

    summary = pd.DataFrame(summary_rows)
    contrast = pd.concat(contrast_frames, ignore_index=True) if contrast_frames else pd.DataFrame()
    family_summary = _family_summary(contrast, min_best_auc=float(strong_auc_threshold))

    paths = {
        "summary": output_dir / "support_path_proxy_feature_gap_summary.csv",
        "feature_contrast": output_dir / "support_path_proxy_feature_gap_contrast.csv",
        "family_summary": output_dir / "support_path_proxy_feature_gap_family_summary.csv",
        "manifest": output_dir / "manifest.json",
    }
    summary.to_csv(paths["summary"], index=False)
    contrast.to_csv(paths["feature_contrast"], index=False)
    family_summary.to_csv(paths["family_summary"], index=False)

    manifest = {
        "scope": "support_path_proxy_feature_gap",
        "labels_path": str(labels_path),
        "output_dir": str(output_dir),
        "rows": int(len(frame)),
        "timestamp_min": frame["__ts__"].min(),
        "timestamp_max": frame["__ts__"].max(),
        "symbols": int(frame["__symbol__"].nunique(dropna=True)),
        "feature_dir": str(feature_dir),
        "feature_list_csv": str(feature_list_csv),
        "max_feature_store_features": max_feature_store_features,
        "max_feature_columns": max_feature_columns,
        "feature_count": int(len(features)),
        "months": [str(v) for v in months],
        "arms": [arm.name for arm in selected_arms],
        "top_fracs": [float(v) for v in top_fracs],
        "selectors": [str(v) for v in selectors],
        "match_modes": [str(v) for v in match_modes],
        "proxy_top_k": int(proxy_top_k),
        "min_train_rows": int(min_train_rows),
        "min_valid_rows": int(min_valid_rows),
        "min_class_rows": int(min_class_rows),
        "strong_auc_threshold": float(strong_auc_threshold),
        "include_causal_time_edge_priors": bool(include_causal_time_edge_priors),
        "include_causal_outcome_priors": bool(include_causal_outcome_priors),
        "include_causal_state_path_priors": bool(include_causal_state_path_priors),
        "include_event_confirmation_features": bool(include_event_confirmation_features),
        "include_adverse_path_composites": bool(include_adverse_path_composites),
        "prior_windows_days": [float(v) for v in prior_windows_days],
        "prior_embargo_hours": float(prior_embargo_hours),
        "outputs": {key: str(value) for key, value in paths.items()},
        "reports": reports,
    }
    markdown = _write_markdown(
        output_dir=output_dir,
        summary=summary,
        contrast=contrast,
        family_summary=family_summary,
        manifest={**manifest, "outputs": {**manifest["outputs"], "markdown": str(output_dir / "support_path_proxy_feature_gap.md")}},
    )
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
    parser.add_argument("--max-feature-columns", type=int, default=0)
    parser.add_argument("--months", type=lambda value: _parse_csv(value, DEFAULT_MONTHS), default=list(DEFAULT_MONTHS))
    parser.add_argument("--arms", type=lambda value: _parse_csv(value, DEFAULT_ARMS), default=list(DEFAULT_ARMS))
    parser.add_argument("--top-fracs", type=lambda value: _parse_float_csv(value), default=list(DEFAULT_TOP_FRACS))
    parser.add_argument("--selectors", type=lambda value: _parse_csv(value, DEFAULT_SELECTORS), default=list(DEFAULT_SELECTORS))
    parser.add_argument("--match-modes", type=lambda value: _parse_csv(value, DEFAULT_MATCH_MODES), default=list(DEFAULT_MATCH_MODES))
    parser.add_argument("--proxy-top-k", type=int, default=DEFAULT_PROXY_TOP_K)
    parser.add_argument("--min-train-rows", type=int, default=250)
    parser.add_argument("--min-valid-rows", type=int, default=25)
    parser.add_argument("--min-class-rows", type=int, default=8)
    parser.add_argument("--strong-auc-threshold", type=float, default=0.70)
    parser.add_argument("--include-causal-time-edge-priors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-causal-outcome-priors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-causal-state-path-priors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-event-confirmation-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-adverse-path-composites", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prior-windows-days", type=lambda value: _parse_float_csv(value), default=list(DEFAULT_PRIOR_WINDOWS_DAYS))
    parser.add_argument("--prior-embargo-hours", type=float, default=24.0)
    parser.add_argument(
        "--state-path-prior-features",
        type=lambda value: _parse_csv(value, tuple(DEFAULT_STATE_PATH_PRIOR_FEATURES)),
        default=list(DEFAULT_STATE_PATH_PRIOR_FEATURES),
    )
    parser.add_argument(
        "--event-feature-store-features",
        type=lambda value: _parse_csv(value, tuple(DEFAULT_EVENT_FEATURE_STORE_FEATURES)),
        default=list(DEFAULT_EVENT_FEATURE_STORE_FEATURES),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_report(
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        max_feature_store_features=args.max_feature_store_features,
        max_feature_columns=args.max_feature_columns,
        months=list(args.months),
        arms=list(args.arms),
        top_fracs=list(args.top_fracs),
        selectors=list(args.selectors),
        match_modes=list(args.match_modes),
        proxy_top_k=int(args.proxy_top_k),
        min_train_rows=int(args.min_train_rows),
        min_valid_rows=int(args.min_valid_rows),
        min_class_rows=int(args.min_class_rows),
        strong_auc_threshold=float(args.strong_auc_threshold),
        include_causal_time_edge_priors=bool(args.include_causal_time_edge_priors),
        include_causal_outcome_priors=bool(args.include_causal_outcome_priors),
        include_causal_state_path_priors=bool(args.include_causal_state_path_priors),
        include_event_confirmation_features=bool(args.include_event_confirmation_features),
        include_adverse_path_composites=bool(args.include_adverse_path_composites),
        prior_windows_days=list(args.prior_windows_days),
        prior_embargo_hours=float(args.prior_embargo_hours),
        state_path_prior_features=list(args.state_path_prior_features),
        event_feature_store_features=list(args.event_feature_store_features),
    )
    print(json.dumps(_json_safe({"output_dir": manifest["output_dir"], "outputs": manifest["outputs"]}), indent=2))


if __name__ == "__main__":
    main()
