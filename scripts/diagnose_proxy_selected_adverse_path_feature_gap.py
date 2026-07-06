#!/usr/bin/env python3
"""Diagnose adverse-path separability inside proxy-selected rows.

This is a no-training diagnostic. It reconstructs causal OOS proxy selections,
then compares selected profitable bounded paths against selected profitable
dirty paths. The intent is to explain why proxy-positive rows can be profitable
in aggregate while still failing MAE/path-risk limits.
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
    _build_frame,
    _bucket_key,
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
from scripts.run_label_adverse_path_proxy_gate_ablation import (  # noqa: E402
    _top_mask,
)
from scripts.run_label_economic_proxy_ablation import (  # noqa: E402
    _economic_targets,
    _label_targets,
)
from scripts.run_label_quality_proxy_diagnostics import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    _feature_columns,
    _json_safe,
    _path_metrics,
    _safe_mean,
    _safe_quantile,
    _selection_metrics,
    _spearman,
)
from scripts.run_soft_label_economic_proxy_ablation import (  # noqa: E402
    DEFAULT_EVENT_FEATURE_STORE_FEATURES,
    DEFAULT_PRIOR_WINDOWS_DAYS,
    DEFAULT_STATE_PATH_PRIOR_FEATURES,
)


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/proxy_selected_adverse_path_feature_gap_v1")
DEFAULT_MONTHS = ("2026-04", "2026-05", "2026-06")
DEFAULT_LABEL_ARMS = (
    "S61_tpnet_strict_adverse_veto_rank",
    "S62_tpnet_clean_dirty_contrast_rank",
    "S65_profit_inside_exec_admissible",
)
DEFAULT_ECONOMIC_ARMS = ("raw_u_policy_net", "risk_u_mild")
DEFAULT_SELECTORS = (
    "label_ic_proxy_oos",
    "combined_l0.50_label_economic_proxy_oos",
)
DEFAULT_TOP_FRACS = (0.03, 0.05)
DEFAULT_MATCH_MODES = ("day_side", "regime_side")


def _parse_csv(value: str | list[str] | tuple[str, ...], default: tuple[str, ...] = ()) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    if not text:
        return list(default)
    return [part.strip() for part in text.split(",") if part.strip()]


def _parse_float_csv(value: str | list[float] | tuple[float, ...]) -> list[float]:
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def _safe_numeric(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _selector_specs(
    *,
    label_arm: str,
    label_score: pd.Series,
    label_features: list[str],
    economic_scores: dict[str, pd.Series],
    economic_features: dict[str, list[str]],
    combine_label_weight: float,
    economic_gate_frac: float,
    requested_selectors: set[str],
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    if "label_ic_proxy_oos" in requested_selectors:
        specs.append(
            {
                "selector": "label_ic_proxy_oos",
                "label_arm": label_arm,
                "economic_arm": "none",
                "score": label_score,
                "proxy_features": label_features,
            }
        )
    combined_name = f"combined_l{combine_label_weight:.2f}_label_economic_proxy_oos"
    gate_name = f"econ_gate{economic_gate_frac:.2f}_then_label_proxy_oos"
    for economic_arm, economic_score in economic_scores.items():
        if combined_name in requested_selectors:
            specs.append(
                {
                    "selector": combined_name,
                    "label_arm": label_arm,
                    "economic_arm": economic_arm,
                    "score": combine_label_weight * label_score + (1.0 - combine_label_weight) * economic_score,
                    "proxy_features": list(dict.fromkeys(label_features + economic_features.get(economic_arm, []))),
                }
            )
        if gate_name in requested_selectors:
            specs.append(
                {
                    "selector": gate_name,
                    "label_arm": label_arm,
                    "economic_arm": economic_arm,
                    "score": label_score.where(_top_gate(economic_score, economic_gate_frac)),
                    "proxy_features": list(dict.fromkeys(label_features + economic_features.get(economic_arm, []))),
                }
            )
    return specs


def _selected_path_masks(metrics: pd.DataFrame, selected: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    mfe_mae = _mfe_mae(metrics)
    profitable = _safe_numeric(metrics["u_policy_net"]).gt(0.0)
    timeout = metrics["is_timeout"].astype(bool)
    bounded = (
        selected
        & profitable
        & _safe_numeric(metrics["mae_norm"]).le(1.0)
        & _safe_numeric(metrics["barrier"]).le(0.025)
        & mfe_mae.ge(1.25)
        & (~timeout)
    )
    dirty = (
        selected
        & profitable
        & (
            _safe_numeric(metrics["mae_norm"]).ge(1.0)
            | _safe_numeric(metrics["barrier"]).gt(0.025)
            | mfe_mae.lt(1.25)
            | timeout
        )
    )
    return profitable & selected, bounded.fillna(False), dirty.fillna(False)


def _select_contrast_features(
    *,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: list[str],
    target_train: pd.Series,
    score: pd.Series,
    proxy_features: list[str],
    max_features: int | None,
) -> list[str]:
    if max_features is None or int(max_features) <= 0 or int(max_features) >= len(features):
        return list(features)
    proxy_set = set(proxy_features)
    priority_families = {
        "adverse_path_composite",
        "event_confirmation",
        "state_path_prior",
        "outcome_prior",
        "liquidity_spread",
        "barrier_distance",
        "pullback_location",
        "exhaustion_reversal",
        "trend_quality",
        "open_interest",
        "chop_compression",
    }
    rows: list[tuple[float, str]] = []
    for feature in features:
        family = _feature_family(feature)
        train_ic = _spearman(train[feature], target_train)
        score_ic = _spearman(valid[feature], score)
        value = 0.0
        if math.isfinite(train_ic):
            value += abs(float(train_ic))
        if math.isfinite(score_ic):
            value += 0.5 * abs(float(score_ic))
        if family in priority_families:
            value += 0.01
        if feature in proxy_set:
            value += 1.0
        rows.append((float(value), feature))
    selected: list[str] = []
    for feature in proxy_features:
        if feature in features and feature not in selected:
            selected.append(feature)
    for _, feature in sorted(rows, key=lambda item: item[0], reverse=True):
        if feature not in selected:
            selected.append(feature)
        if len(selected) >= int(max_features):
            break
    return selected


def _summary_row(
    *,
    valid: pd.DataFrame,
    valid_metrics: pd.DataFrame,
    target_valid: pd.DataFrame,
    score: pd.Series,
    selected: pd.Series,
    profitable_selected: pd.Series,
    bounded_selected: pd.Series,
    dirty_selected: pd.Series,
    matched_bounded: pd.Series,
    matched_dirty: pd.Series,
    eligible_buckets: int,
    month: str,
    selector: str,
    label_arm: str,
    economic_arm: str,
    top_frac: float,
    match_mode: str,
    economic_target: pd.Series | None,
    proxy_features: list[str],
    top_contrast: pd.DataFrame,
) -> dict[str, Any]:
    metrics = _selection_metrics(
        frame=valid,
        metrics=valid_metrics,
        target=target_valid,
        score=score,
        arm=f"{selector}::{label_arm}::{economic_arm}",
        selector=selector,
        period=month,
        top_frac=top_frac,
    )
    row = {
        "month": month,
        "selector": selector,
        "label_arm": label_arm,
        "economic_arm": economic_arm,
        "top_frac": float(top_frac),
        "match_mode": match_mode,
        "valid_rows": int(len(valid)),
        "selected_rows": int(selected.sum()),
        "profitable_selected_rows": int(profitable_selected.sum()),
        "bounded_selected_rows": int(bounded_selected.sum()),
        "dirty_selected_rows": int(dirty_selected.sum()),
        "bounded_selected_rate": float(bounded_selected.sum() / selected.sum()) if int(selected.sum()) else 0.0,
        "dirty_selected_rate": float(dirty_selected.sum() / selected.sum()) if int(selected.sum()) else 0.0,
        "matched_bounded_rows": int(matched_bounded.sum()),
        "matched_dirty_rows": int(matched_dirty.sum()),
        "matched_bucket_count": int(eligible_buckets),
        "mean_return_net": metrics.get("mean_return_net"),
        "bad_mae_1r_rate": metrics.get("bad_mae_1r_rate"),
        "p90_mae_norm": metrics.get("p90_mae_norm"),
        "wide_barrier_25bps_rate": metrics.get("wide_barrier_25bps_rate"),
        "timeout_rate": metrics.get("timeout_rate"),
        "strict_clean_row_rate": metrics.get("strict_clean_row_rate"),
        "score_ic_u": _spearman(score, valid_metrics["u_policy_net"]),
        "score_ic_label": _spearman(score, target_valid["target_soft"]),
        "score_ic_economic": (
            _spearman(score, economic_target.reset_index(drop=True))
            if economic_target is not None
            else float("nan")
        ),
        "decile_spearman_u": metrics.get("decile_spearman_u"),
        "top_bottom_decile_spread_u": metrics.get("top_bottom_decile_spread_u"),
        "proxy_features": ",".join(proxy_features),
    }
    if not top_contrast.empty:
        top = top_contrast.iloc[0]
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


def _family_summary(contrast: pd.DataFrame) -> pd.DataFrame:
    if contrast.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for family, group in contrast.groupby("feature_family", dropna=False, observed=True):
        top_features = (
            group.sort_values(["best_auc", "abs_bucket_gap"], ascending=[False, False])["feature"]
            .drop_duplicates()
            .head(8)
            .tolist()
        )
        rows.append(
            {
                "feature_family": family,
                "rows": int(len(group)),
                "months": int(group["month"].nunique()),
                "labels": int(group["label_arm"].nunique()),
                "selectors": int(group["selector"].nunique()),
                "mean_best_auc": _safe_mean(group["best_auc"]),
                "max_best_auc": _safe_quantile(group["best_auc"], 1.0),
                "mean_abs_bucket_gap": _safe_mean(group["abs_bucket_gap"]),
                "proxy_feature_share": _safe_mean(group["is_proxy_feature"].astype(float)),
                "top_features": ",".join(top_features),
            }
        )
    return pd.DataFrame(rows).sort_values(["max_best_auc", "mean_abs_bucket_gap"], ascending=[False, False])


def _write_markdown(
    *,
    output_dir: Path,
    summary: pd.DataFrame,
    contrast: pd.DataFrame,
    family_summary: pd.DataFrame,
    manifest: dict[str, Any],
) -> Path:
    path = output_dir / "proxy_selected_adverse_path_feature_gap.md"
    summary_cols = [
        "month",
        "selector",
        "label_arm",
        "economic_arm",
        "top_frac",
        "match_mode",
        "selected_rows",
        "profitable_selected_rows",
        "bounded_selected_rows",
        "dirty_selected_rows",
        "bounded_selected_rate",
        "dirty_selected_rate",
        "matched_bounded_rows",
        "matched_dirty_rows",
        "mean_return_net",
        "bad_mae_1r_rate",
        "p90_mae_norm",
        "wide_barrier_25bps_rate",
        "timeout_rate",
        "score_ic_u",
        "decile_spearman_u",
        "top_feature",
        "top_feature_family",
        "top_feature_best_auc",
        "top_feature_direction",
        "top_feature_is_proxy_feature",
    ]
    contrast_cols = [
        "month",
        "selector",
        "label_arm",
        "economic_arm",
        "top_frac",
        "match_mode",
        "feature",
        "feature_family",
        "is_proxy_feature",
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
        "labels",
        "selectors",
        "mean_best_auc",
        "max_best_auc",
        "mean_abs_bucket_gap",
        "proxy_feature_share",
        "top_features",
    ]
    lines = [
        "# Proxy-Selected Adverse-Path Feature Gap",
        "",
        "Scope: no model training. Each month is scored by proxies fit only on prior rows, then selected profitable bounded paths are contrasted with selected profitable dirty paths.",
        "",
        f"Labels: `{', '.join(manifest['label_arms'])}`",
        f"Economic arms: `{', '.join(manifest['economic_arms'])}`",
        f"Selectors: `{', '.join(manifest['selectors'])}`",
        f"Months: `{', '.join(manifest['months'])}`",
        f"Top fractions: `{manifest['top_fracs']}`",
        f"Match modes: `{manifest['match_modes']}`",
        f"Feature count: `{manifest['feature_count']}`",
        "",
        "## Summary",
        "",
        _table(
            summary.sort_values(["dirty_selected_rate", "p90_mae_norm"], ascending=[False, False])
            if not summary.empty
            else summary,
            summary_cols,
            limit=80,
        ),
        "",
        "## Feature Families",
        "",
        _table(family_summary, family_cols, limit=40),
        "",
        "## Top Feature Contrasts",
        "",
        _table(
            contrast.sort_values(["best_auc", "abs_bucket_gap"], ascending=[False, False])
            if not contrast.empty
            else contrast,
            contrast_cols,
            limit=100,
        ),
        "",
        "## Outputs",
        "",
        f"- Summary: `{manifest['outputs']['summary']}`",
        f"- Feature contrast: `{manifest['outputs']['contrast']}`",
        f"- Feature family summary: `{manifest['outputs']['family_summary']}`",
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
    label_arms: list[str],
    economic_arms: list[str],
    selectors: list[str],
    months: list[str],
    top_fracs: list[float],
    match_modes: list[str],
    combine_label_weight: float,
    economic_gate_frac: float,
    proxy_top_k: int,
    min_train_rows: int,
    min_valid_rows: int,
    min_class_rows: int,
    max_contrast_features: int | None,
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
    features = _feature_columns(frame)
    targets = _label_targets(frame, metrics)
    economic_targets = _economic_targets(metrics)
    missing_labels = sorted(set(label_arms).difference(targets))
    missing_economic = sorted(set(economic_arms).difference(economic_targets))
    if missing_labels:
        raise ValueError(f"Unknown label arms: {missing_labels}")
    if missing_economic:
        raise ValueError(f"Unknown economic arms: {missing_economic}")

    requested_selectors = set(selectors)
    month_period = frame["__ts__"].dt.to_period("M").astype(str)
    summary_rows: list[dict[str, Any]] = []
    contrast_parts: list[pd.DataFrame] = []

    for month in months:
        train_mask = month_period.lt(str(month))
        valid_mask = month_period.eq(str(month))
        if int(train_mask.sum()) < int(min_train_rows) or int(valid_mask.sum()) < int(min_valid_rows):
            continue
        train = frame.loc[train_mask].copy()
        valid_source = frame.loc[valid_mask].copy()
        valid = valid_source.reset_index(drop=True)
        valid_metrics = metrics.loc[valid_mask].copy().reset_index(drop=True)

        economic_scores: dict[str, pd.Series] = {}
        economic_features: dict[str, list[str]] = {}
        for economic_arm in economic_arms:
            score, diag = _score_proxy(
                train=train,
                valid=valid_source,
                features=features,
                y_train=economic_targets[economic_arm].loc[train_mask],
                proxy_top_k=proxy_top_k,
            )
            economic_scores[economic_arm] = score.reset_index(drop=True)
            economic_features[economic_arm] = list(diag.get("proxy_features", []))

        for label_arm in label_arms:
            target = targets[label_arm]
            target_valid = target.loc[valid_mask].copy().reset_index(drop=True)
            label_score, label_diag = _score_proxy(
                train=train,
                valid=valid_source,
                features=features,
                y_train=target.loc[train_mask, "target_soft"],
                proxy_top_k=proxy_top_k,
            )
            label_score = label_score.reset_index(drop=True)
            specs = _selector_specs(
                label_arm=label_arm,
                label_score=label_score,
                label_features=list(label_diag.get("proxy_features", [])),
                economic_scores=economic_scores,
                economic_features=economic_features,
                combine_label_weight=combine_label_weight,
                economic_gate_frac=economic_gate_frac,
                requested_selectors=requested_selectors,
            )
            for spec in specs:
                score = _safe_numeric(spec["score"]).reset_index(drop=True)
                economic_target = (
                    economic_targets[spec["economic_arm"]].loc[valid_mask].copy().reset_index(drop=True)
                    if spec["economic_arm"] != "none"
                    else None
                )
                for top_frac in top_fracs:
                    selected = _top_mask(score, float(top_frac))
                    profitable_selected, bounded_selected, dirty_selected = _selected_path_masks(
                        valid_metrics,
                        selected,
                    )
                    for match_mode in match_modes:
                        bucket = _bucket_key(valid, valid_metrics, match_mode)
                        matched_bounded, matched_dirty, eligible, eligible_bucket_count = _eligible_masks(
                            bucket=bucket,
                            missed_clean_mask=bounded_selected,
                            dirty_false_positive_mask=dirty_selected,
                        )
                        contrast_features = _select_contrast_features(
                            train=train,
                            valid=valid,
                            features=features,
                            target_train=target.loc[train_mask, "target_soft"],
                            score=score,
                            proxy_features=list(spec["proxy_features"]),
                            max_features=max_contrast_features,
                        )
                        contrast = _feature_contrast(
                            train=train,
                            valid=valid,
                            valid_metrics=valid_metrics,
                            features=contrast_features,
                            target_train=target.loc[train_mask, "target_soft"],
                            target_valid=target_valid["target_soft"],
                            score=score,
                            bucket=bucket,
                            clean_mask=matched_bounded,
                            dirty_mask=matched_dirty,
                            eligible_candidate_mask=eligible,
                            proxy_features=list(spec["proxy_features"]),
                            min_class_rows=min_class_rows,
                            source="all",
                            month=str(month),
                            label_arm=str(label_arm),
                            top_frac=float(top_frac),
                            match_mode=str(match_mode),
                        )
                        if not contrast.empty:
                            contrast["selector"] = spec["selector"]
                            contrast["economic_arm"] = spec["economic_arm"]
                            contrast_parts.append(contrast)
                        summary_rows.append(
                            _summary_row(
                                valid=valid,
                                valid_metrics=valid_metrics,
                                target_valid=target_valid,
                                score=score,
                                selected=selected,
                                profitable_selected=profitable_selected,
                                bounded_selected=bounded_selected,
                                dirty_selected=dirty_selected,
                                matched_bounded=matched_bounded,
                                matched_dirty=matched_dirty,
                                eligible_buckets=eligible_bucket_count,
                                month=str(month),
                                selector=str(spec["selector"]),
                                label_arm=str(label_arm),
                                economic_arm=str(spec["economic_arm"]),
                                top_frac=float(top_frac),
                                match_mode=str(match_mode),
                                economic_target=economic_target,
                                proxy_features=list(spec["proxy_features"]),
                                top_contrast=contrast,
                            )
                        )

    summary = pd.DataFrame(summary_rows)
    contrast = pd.concat(contrast_parts, ignore_index=True) if contrast_parts else pd.DataFrame()
    family_summary = _family_summary(contrast)

    paths = {
        "summary": output_dir / "proxy_selected_adverse_path_summary.csv",
        "contrast": output_dir / "proxy_selected_adverse_path_feature_contrast.csv",
        "family_summary": output_dir / "proxy_selected_adverse_path_feature_family_summary.csv",
        "manifest": output_dir / "manifest.json",
    }
    summary.to_csv(paths["summary"], index=False)
    contrast.to_csv(paths["contrast"], index=False)
    family_summary.to_csv(paths["family_summary"], index=False)
    manifest = {
        "scope": "proxy_selected_adverse_path_feature_gap",
        "labels_path": str(labels_path),
        "output_dir": str(output_dir),
        "rows": int(len(frame)),
        "timestamp_min": frame["__ts__"].min(),
        "timestamp_max": frame["__ts__"].max(),
        "symbols": int(frame["__symbol__"].nunique(dropna=True)),
        "feature_dir": str(feature_dir),
        "feature_list_csv": str(feature_list_csv),
        "max_feature_store_features": max_feature_store_features,
        "label_arms": list(label_arms),
        "economic_arms": list(economic_arms),
        "selectors": list(selectors),
        "months": list(months),
        "top_fracs": [float(v) for v in top_fracs],
        "match_modes": list(match_modes),
        "combine_label_weight": float(combine_label_weight),
        "economic_gate_frac": float(economic_gate_frac),
        "proxy_top_k": int(proxy_top_k),
        "min_train_rows": int(min_train_rows),
        "min_valid_rows": int(min_valid_rows),
        "min_class_rows": int(min_class_rows),
        "max_contrast_features": None if max_contrast_features is None else int(max_contrast_features),
        "include_causal_outcome_priors": bool(include_causal_outcome_priors),
        "include_causal_state_path_priors": bool(include_causal_state_path_priors),
        "include_event_confirmation_features": bool(include_event_confirmation_features),
        "include_adverse_path_composites": bool(include_adverse_path_composites),
        "prior_windows_days": [float(v) for v in prior_windows_days],
        "prior_embargo_hours": float(prior_embargo_hours),
        "feature_count": int(len(features)),
        "reports": reports,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    markdown = _write_markdown(
        output_dir=output_dir,
        summary=summary,
        contrast=contrast,
        family_summary=family_summary,
        manifest={**manifest, "outputs": {**manifest["outputs"], "markdown": str(output_dir / "proxy_selected_adverse_path_feature_gap.md")}},
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
    parser.add_argument("--max-feature-store-features", type=int, default=None)
    parser.add_argument("--label-arms", type=lambda value: _parse_csv(value, DEFAULT_LABEL_ARMS), default=list(DEFAULT_LABEL_ARMS))
    parser.add_argument("--economic-arms", type=lambda value: _parse_csv(value, DEFAULT_ECONOMIC_ARMS), default=list(DEFAULT_ECONOMIC_ARMS))
    parser.add_argument("--selectors", type=lambda value: _parse_csv(value, DEFAULT_SELECTORS), default=list(DEFAULT_SELECTORS))
    parser.add_argument("--months", type=lambda value: _parse_csv(value, DEFAULT_MONTHS), default=list(DEFAULT_MONTHS))
    parser.add_argument("--top-fracs", type=lambda value: _parse_float_csv(value), default=list(DEFAULT_TOP_FRACS))
    parser.add_argument("--match-modes", type=lambda value: _parse_csv(value, DEFAULT_MATCH_MODES), default=list(DEFAULT_MATCH_MODES))
    parser.add_argument("--combine-label-weight", type=float, default=0.50)
    parser.add_argument("--economic-gate-frac", type=float, default=0.30)
    parser.add_argument("--proxy-top-k", type=int, default=DEFAULT_PROXY_TOP_K)
    parser.add_argument("--min-train-rows", type=int, default=500)
    parser.add_argument("--min-valid-rows", type=int, default=100)
    parser.add_argument("--min-class-rows", type=int, default=5)
    parser.add_argument("--max-contrast-features", type=int, default=300)
    parser.add_argument("--include-causal-outcome-priors", action="store_true")
    parser.add_argument("--include-causal-state-path-priors", action="store_true")
    parser.add_argument("--include-event-confirmation-features", action="store_true")
    parser.add_argument("--include-adverse-path-composites", action="store_true")
    parser.add_argument("--prior-windows-days", type=lambda value: _parse_float_csv(value), default=list(DEFAULT_PRIOR_WINDOWS_DAYS))
    parser.add_argument("--prior-embargo-hours", type=float, default=24.0)
    parser.add_argument(
        "--state-path-prior-features",
        type=lambda value: _parse_csv(value, DEFAULT_STATE_PATH_PRIOR_FEATURES),
        default=list(DEFAULT_STATE_PATH_PRIOR_FEATURES),
    )
    parser.add_argument(
        "--event-feature-store-features",
        type=lambda value: _parse_csv(value, DEFAULT_EVENT_FEATURE_STORE_FEATURES),
        default=list(DEFAULT_EVENT_FEATURE_STORE_FEATURES),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_report(
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        max_feature_store_features=args.max_feature_store_features,
        label_arms=list(args.label_arms),
        economic_arms=list(args.economic_arms),
        selectors=list(args.selectors),
        months=list(args.months),
        top_fracs=list(args.top_fracs),
        match_modes=list(args.match_modes),
        combine_label_weight=float(args.combine_label_weight),
        economic_gate_frac=float(args.economic_gate_frac),
        proxy_top_k=int(args.proxy_top_k),
        min_train_rows=int(args.min_train_rows),
        min_valid_rows=int(args.min_valid_rows),
        min_class_rows=int(args.min_class_rows),
        max_contrast_features=None if args.max_contrast_features is None else int(args.max_contrast_features),
        include_causal_outcome_priors=bool(args.include_causal_outcome_priors),
        include_causal_state_path_priors=bool(args.include_causal_state_path_priors),
        include_event_confirmation_features=bool(args.include_event_confirmation_features),
        include_adverse_path_composites=bool(args.include_adverse_path_composites),
        prior_windows_days=list(args.prior_windows_days),
        prior_embargo_hours=float(args.prior_embargo_hours),
        state_path_prior_features=list(args.state_path_prior_features),
        event_feature_store_features=list(args.event_feature_store_features),
    )
    print(
        json.dumps(
            _json_safe({key: value for key, value in manifest.items() if key not in {"reports"}}),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
