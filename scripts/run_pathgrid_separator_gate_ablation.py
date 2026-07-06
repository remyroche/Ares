#!/usr/bin/env python3
"""Causal separator-gate ablation for Stage108 path-grid labels.

This is a proxy-only label test. It does not train LightGBM, run Optuna, or
optimise policy geometry. It asks whether prior OOS clean-vs-dirty separator
features can make the Stage108 economic proxy select rows that are both
learnable and inside the execution envelope.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnose_label_matched_clean_dirty_feature_gap import (  # noqa: E402
    _bucket_key,
    _rank_within_bucket,
)
from scripts.diagnose_stage108_pathgrid_recoverability import (  # noqa: E402
    _choose_candidates,
    _dirty,
    _spec_from_row,
    _strict_clean,
)
from scripts.report_label_proxy_quality_with_economic_limits import (  # noqa: E402
    _fit_holdout_summary,
    _score_period,
    _slice_week_positions,
    _table,
)
from scripts.run_label_quality_proxy_diagnostics import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    _feature_columns,
    _json_safe,
    _rank_top_indices,
    _safe_mean,
    _safe_quantile,
    _spearman,
)
from scripts.run_path_aware_label_target_grid import (  # noqa: E402
    DEFAULT_FEATURE_LIST_CSV,
    _target_for_spec,
)
from scripts.run_soft_label_economic_proxy_ablation import (  # noqa: E402
    DEFAULT_EVENT_FEATURE_STORE_FEATURES,
    DEFAULT_PRIOR_WINDOWS_DAYS,
    DEFAULT_STATE_PATH_PRIOR_FEATURES,
    _proxy_score,
)
from scripts.diagnose_label_matched_clean_dirty_feature_gap import _build_frame  # noqa: E402


DEFAULT_STAGE_DIR = Path("data_perp/reports/path_aware_label_target_grid_stage108_decisive_hard_econic_v1")
DEFAULT_RECOVERABILITY_DIR = Path("data_perp/reports/pathgrid_stage108_recoverability_stage109_v1")
DEFAULT_OUTPUT_DIR = Path("data_perp/reports/pathgrid_separator_gate_stage110_v1")
DEFAULT_MONTHS = ("2026-04", "2026-05", "2026-06")
DEFAULT_FIT_MONTHS = ("2026-05",)
DEFAULT_HOLDOUT_MONTH = "2026-06"
DEFAULT_TOP_FRACS = (0.005, 0.01)
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


def _rank_pct(values: Any) -> pd.Series:
    series = _safe_numeric(values)
    if int(series.notna().sum()) < 2:
        return pd.Series(np.nan, index=series.index, dtype=np.float32)
    return series.rank(method="average", pct=True).astype(np.float32)


def _top_mask(score: pd.Series, frac: float) -> pd.Series:
    mask = pd.Series(False, index=score.index)
    idx = _rank_top_indices(score, frac)
    if len(idx):
        mask.iloc[idx] = True
    return mask


def _month_lt(left: Any, right: str) -> pd.Series:
    left_period = pd.PeriodIndex(pd.Series(left).astype(str), freq="M")
    return pd.Series(left_period < pd.Period(str(right), freq="M"))


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    vals = _safe_numeric(values)
    w = _safe_numeric(weights).fillna(0.0)
    mask = vals.notna() & w.gt(0.0)
    if not bool(mask.any()):
        return float("nan")
    return float(np.average(vals[mask], weights=w[mask]))


def _select_separator_features(
    contrast: pd.DataFrame,
    *,
    candidate: str,
    month: str,
    top_frac: float,
    match_mode: str,
    max_features: int,
    max_per_family: int,
    min_auc: float,
    min_abs_gap: float,
    min_direction_consistency: float,
) -> pd.DataFrame:
    if contrast.empty:
        return pd.DataFrame()
    work = contrast[
        contrast["label_arm"].astype(str).eq(str(candidate))
        & _month_lt(contrast["month"], str(month)).to_numpy()
        & pd.to_numeric(contrast["top_frac"], errors="coerce").sub(float(top_frac)).abs().le(1e-12)
        & contrast["match_mode"].astype(str).eq(str(match_mode))
    ].copy()
    if work.empty:
        return pd.DataFrame()
    work["best_auc"] = pd.to_numeric(work["best_auc"], errors="coerce")
    work["abs_bucket_gap"] = pd.to_numeric(work["abs_bucket_gap"], errors="coerce")
    work = work[work["best_auc"].ge(float(min_auc)) & work["abs_bucket_gap"].ge(float(min_abs_gap))].copy()
    if work.empty:
        return pd.DataFrame()
    work["edge"] = (work["best_auc"] - 0.5).clip(lower=0.0) * work["abs_bucket_gap"].clip(lower=0.0)
    work = work[work["edge"].gt(0.0)].copy()
    if work.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for feature, group in work.groupby("feature", dropna=False):
        direction_weights = group.groupby("best_direction", dropna=False)["edge"].sum()
        if direction_weights.empty:
            continue
        best_direction = str(direction_weights.sort_values(ascending=False).index[0])
        total_edge = float(direction_weights.sum())
        direction_edge = float(direction_weights.loc[best_direction])
        consistency = direction_edge / total_edge if total_edge > 0.0 else 0.0
        if consistency < float(min_direction_consistency):
            continue
        best_rows = group[group["best_direction"].astype(str).eq(best_direction)].copy()
        rows.append(
            {
                "feature": str(feature),
                "feature_family": str(best_rows["feature_family"].mode(dropna=True).iloc[0])
                if "feature_family" in best_rows and not best_rows["feature_family"].mode(dropna=True).empty
                else "unknown",
                "best_direction": best_direction,
                "edge_sum": total_edge,
                "direction_edge": direction_edge,
                "direction_consistency": consistency,
                "max_best_auc": float(best_rows["best_auc"].max()),
                "mean_best_auc": _weighted_mean(best_rows["best_auc"], best_rows["edge"]),
                "mean_abs_bucket_gap": _weighted_mean(best_rows["abs_bucket_gap"], best_rows["edge"]),
                "prior_months": ",".join(sorted(best_rows["month"].astype(str).unique())),
                "prior_rows": int(len(best_rows)),
            }
        )
    selected = pd.DataFrame(rows)
    if selected.empty:
        return selected
    selected = selected.sort_values(
        ["edge_sum", "max_best_auc", "mean_abs_bucket_gap"],
        ascending=[False, False, False],
    )
    kept: list[int] = []
    family_counts: dict[str, int] = {}
    for idx, row in selected.iterrows():
        family = str(row["feature_family"])
        if family_counts.get(family, 0) >= int(max_per_family):
            continue
        kept.append(idx)
        family_counts[family] = family_counts.get(family, 0) + 1
        if len(kept) >= int(max_features):
            break
    return selected.loc[kept].reset_index(drop=True)


def _separator_score(
    *,
    valid: pd.DataFrame,
    valid_metrics: pd.DataFrame,
    selected_features: pd.DataFrame,
    match_mode: str,
) -> pd.Series:
    if selected_features.empty:
        return pd.Series(np.nan, index=valid.index, dtype=np.float32)
    bucket = _bucket_key(valid, valid_metrics, match_mode).reset_index(drop=True)
    parts: list[pd.Series] = []
    for _, row in selected_features.iterrows():
        feature = str(row["feature"])
        if feature not in valid.columns:
            continue
        ranks = _rank_within_bucket(_safe_numeric(valid[feature]).reset_index(drop=True), bucket).astype(np.float32)
        if str(row["best_direction"]) == "clean_low":
            ranks = 1.0 - ranks
        parts.append(ranks.fillna(0.5))
    if not parts:
        return pd.Series(np.nan, index=valid.index, dtype=np.float32)
    return pd.concat(parts, axis=1).mean(axis=1).astype(np.float32)


def _selector_scores(proxy_score: pd.Series, separator_score: pd.Series) -> dict[str, pd.Series]:
    proxy_rank = _rank_pct(proxy_score).fillna(0.5)
    sep_rank = _rank_pct(separator_score)
    scores: dict[str, pd.Series] = {
        "economic_ic_proxy_oos": proxy_score.reset_index(drop=True),
    }
    if int(sep_rank.notna().sum()) < 2:
        return scores
    sep_rank = sep_rank.fillna(0.5)
    scores["separator_only_prior"] = sep_rank
    scores["proxy_sep_blend_70_30"] = (0.70 * proxy_rank + 0.30 * sep_rank).astype(np.float32)
    scores["proxy_sep_blend_50_50"] = (0.50 * proxy_rank + 0.50 * sep_rank).astype(np.float32)
    scores["proxy_x_separator"] = (proxy_rank * sep_rank).astype(np.float32)
    for keep_frac in (0.70, 0.50, 0.30):
        threshold = 1.0 - float(keep_frac)
        gated = proxy_rank.copy()
        gated[sep_rank < threshold] = -1.0 + 0.001 * proxy_rank[sep_rank < threshold]
        scores[f"proxy_gate_separator_keep{int(round(keep_frac * 100)):02d}"] = gated.astype(np.float32)
    return scores


def _selected_ledger_rows(
    *,
    valid: pd.DataFrame,
    valid_metrics: pd.DataFrame,
    valid_target: pd.DataFrame,
    candidate: str,
    month: str,
    selector: str,
    top_frac: float,
    score: pd.Series,
    separator_features: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    idx = _rank_top_indices(score, float(top_frac))
    if not len(idx):
        return rows
    mfe_mae = (
        valid_metrics["mfe_norm"] / valid_metrics["mae_norm"].clip(lower=0.25)
    ).replace([np.inf, -np.inf], np.nan).clip(upper=10.0)
    selected_features = ",".join(separator_features["feature"].astype(str).tolist()) if not separator_features.empty else ""
    for pos in idx:
        rows.append(
            {
                "candidate": candidate,
                "month": str(month),
                "selector": selector,
                "top_frac": float(top_frac),
                "position": int(pos),
                "__ts__": valid["__ts__"].iloc[pos],
                "__symbol__": valid["__symbol__"].iloc[pos],
                "side": valid_metrics["side"].iloc[pos],
                "score": score.iloc[pos],
                "target_soft": valid_target["target_soft"].iloc[pos],
                "target_hard": valid_target["target_hard"].iloc[pos],
                "u_policy_net": valid_metrics["u_policy_net"].iloc[pos],
                "ret_net": valid_metrics["ret_net"].iloc[pos],
                "mae_norm": valid_metrics["mae_norm"].iloc[pos],
                "mfe_norm": valid_metrics["mfe_norm"].iloc[pos],
                "mfe_mae": mfe_mae.iloc[pos],
                "barrier": valid_metrics["barrier"].iloc[pos],
                "is_timeout": bool(valid_metrics["is_timeout"].iloc[pos]),
                "bars_to_mfe": valid_metrics["bars_to_mfe"].iloc[pos],
                "separator_features": selected_features,
            }
        )
    return rows


def _month_period_rows(
    *,
    valid: pd.DataFrame,
    valid_metrics: pd.DataFrame,
    valid_target: pd.DataFrame,
    candidate: str,
    month: str,
    top_fracs: list[float],
    selector_scores: dict[str, pd.Series],
    selector_features: dict[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    period_slices = [("month", str(month), np.arange(len(valid), dtype=np.int64))]
    period_slices.extend(("week", week, pos) for week, pos in _slice_week_positions(valid))
    for selector, score in selector_scores.items():
        score = _safe_numeric(score).reset_index(drop=True)
        for period_type, period_name, pos in period_slices:
            for frac in top_fracs:
                rows.append(
                    _score_period(
                        frame=valid.iloc[pos].reset_index(drop=True),
                        metrics=valid_metrics.iloc[pos].reset_index(drop=True),
                        target=valid_target.iloc[pos].reset_index(drop=True),
                        score=score.iloc[pos].reset_index(drop=True),
                        period_type=period_type,
                        period=period_name,
                        month=str(month),
                        selector=selector,
                        label_arm=candidate,
                        economic_arm="pathgrid_separator_gate",
                        top_frac=float(frac),
                        label_score=valid_target["target_soft"].iloc[pos].reset_index(drop=True),
                        economic_score=None,
                        economic_target=None,
                        label_proxy_features=selector_features.get(selector, ""),
                        economic_proxy_features="",
                    )
                )
    return rows


def _confusion_row(
    *,
    valid_metrics: pd.DataFrame,
    valid_target: pd.DataFrame,
    candidate: str,
    month: str,
    selector: str,
    top_frac: float,
    score: pd.Series,
    separator_features: pd.DataFrame,
) -> dict[str, Any]:
    oracle = _top_mask(valid_target["target_soft"], float(top_frac)).reset_index(drop=True)
    selected = _top_mask(score, float(top_frac)).reset_index(drop=True)
    strict_clean = _strict_clean(valid_metrics).reset_index(drop=True)
    dirty = _dirty(valid_metrics).reset_index(drop=True)
    recovered = selected & oracle
    clean_selected = selected & strict_clean
    dirty_selected = selected & dirty
    selected_metrics = valid_metrics.loc[selected].copy()
    selected_target = valid_target.loc[selected].copy()
    return {
        "candidate": candidate,
        "month": str(month),
        "selector": selector,
        "top_frac": float(top_frac),
        "selected_rows": int(selected.sum()),
        "oracle_recovery_rate": float(recovered.sum() / max(int(oracle.sum()), 1)),
        "target_hard_rate": _safe_mean(selected_target["target_hard"]) if int(selected.sum()) else float("nan"),
        "strict_clean_rate": _safe_mean(clean_selected.loc[selected].astype(float)) if int(selected.sum()) else float("nan"),
        "dirty_rate": _safe_mean(dirty_selected.loc[selected].astype(float)) if int(selected.sum()) else float("nan"),
        "mean_return_net": _safe_mean(selected_metrics["ret_net"]) if int(selected.sum()) else float("nan"),
        "mean_u": _safe_mean(selected_metrics["u_policy_net"]) if int(selected.sum()) else float("nan"),
        "bad_mae_1r_rate": _safe_mean((selected_metrics["mae_norm"] >= 1.0).astype(float))
        if int(selected.sum())
        else float("nan"),
        "timeout_rate": _safe_mean(selected_metrics["is_timeout"].astype(float)) if int(selected.sum()) else float("nan"),
        "p90_mae_norm": _safe_quantile(selected_metrics["mae_norm"], 0.90) if int(selected.sum()) else float("nan"),
        "score_ic_u": _spearman(score, valid_metrics["u_policy_net"]),
        "score_ic_label": _spearman(score, valid_target["target_soft"]),
        "separator_feature_count": int(len(separator_features)),
        "separator_features": ",".join(separator_features["feature"].astype(str).tolist())
        if not separator_features.empty
        else "",
    }


def _candidate_summary(fit_holdout: pd.DataFrame) -> pd.DataFrame:
    if fit_holdout.empty:
        return fit_holdout
    return fit_holdout.sort_values(
        [
            "trainworthy_pass",
            "holdout_economic_pass",
            "fit_economic_pass",
            "holdout_mean_return_net",
            "holdout_bad_mae_1r_rate",
            "holdout_timeout_rate",
        ],
        ascending=[False, False, False, False, True, True],
    ).reset_index(drop=True)


def _write_report(
    *,
    output_dir: Path,
    fit_holdout: pd.DataFrame,
    period_rows: pd.DataFrame,
    confusion: pd.DataFrame,
    separator_features: pd.DataFrame,
    manifest: dict[str, Any],
) -> Path:
    path = output_dir / "pathgrid_separator_gate_ablation.md"
    monthly = period_rows[period_rows["period_type"].astype(str).eq("month")].copy() if not period_rows.empty else pd.DataFrame()
    lines = [
        "# Path-Grid Separator Gate Ablation",
        "",
        "Scope: proxy-only causal gate on Stage108 hard-binary path-grid labels. No model training, Optuna, or policy optimisation.",
        "",
        f"Stage dir: `{manifest['stage_dir']}`",
        f"Recoverability dir: `{manifest['recoverability_dir']}`",
        f"Months: `{', '.join(manifest['months'])}`. Fit months: `{', '.join(manifest['fit_months'])}`. Holdout: `{manifest['holdout_month']}`.",
        f"Candidates: `{', '.join(manifest['candidates'])}`",
        f"Top fractions: `{manifest['top_fracs']}`. Match modes: `{manifest['match_modes']}`.",
        "",
        "## Fit/Holdout Summary",
        "",
        _table(
            fit_holdout,
            [
                "trainworthy_pass",
                "fit_economic_pass",
                "holdout_economic_pass",
                "selector",
                "label_arm",
                "top_frac",
                "fit_mean_return_net",
                "holdout_mean_return_net",
                "fit_bad_mae_1r_rate",
                "holdout_bad_mae_1r_rate",
                "fit_p90_mae_norm",
                "holdout_p90_mae_norm",
                "fit_timeout_rate",
                "holdout_timeout_rate",
                "fit_score_ic_u",
                "holdout_score_ic_u",
                "fit_selected_rows",
                "holdout_selected_rows",
            ],
            limit=80,
        ),
        "",
        "## Monthly OOS Rows",
        "",
        _table(
            monthly.sort_values(["month", "label_arm", "top_frac", "selector"]) if not monthly.empty else monthly,
            [
                "month",
                "selector",
                "label_arm",
                "top_frac",
                "selected_rows",
                "mean_return_net",
                "bad_mae_1r_rate",
                "p90_mae_norm",
                "timeout_rate",
                "wide_barrier_25bps_rate",
                "score_ic_u",
                "score_ic_label",
                "strict_clean_row_rate",
                "target_top_hard_rate",
            ],
            limit=140,
        ),
        "",
        "## Selector Confusion",
        "",
        _table(
            confusion.sort_values(["month", "candidate", "top_frac", "selector"]) if not confusion.empty else confusion,
            [
                "month",
                "candidate",
                "selector",
                "top_frac",
                "selected_rows",
                "oracle_recovery_rate",
                "target_hard_rate",
                "strict_clean_rate",
                "dirty_rate",
                "mean_return_net",
                "bad_mae_1r_rate",
                "timeout_rate",
                "p90_mae_norm",
                "score_ic_u",
                "separator_feature_count",
            ],
            limit=160,
        ),
        "",
        "## Prior Separator Features",
        "",
        _table(
            separator_features.sort_values(["month", "candidate", "top_frac", "match_mode", "rank"])
            if not separator_features.empty
            else separator_features,
            [
                "month",
                "candidate",
                "top_frac",
                "match_mode",
                "rank",
                "feature",
                "feature_family",
                "best_direction",
                "edge_sum",
                "direction_consistency",
                "max_best_auc",
                "mean_abs_bucket_gap",
                "prior_months",
            ],
            limit=120,
        ),
        "",
        "## Outputs",
        "",
        f"- Period rows: `{manifest['outputs']['period_rows']}`",
        f"- Fit/holdout: `{manifest['outputs']['fit_holdout']}`",
        f"- Confusion: `{manifest['outputs']['confusion']}`",
        f"- Separator features: `{manifest['outputs']['separator_features']}`",
        f"- Selected ledger: `{manifest['outputs']['selected_ledger']}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_report(
    *,
    stage_dir: Path,
    recoverability_dir: Path,
    output_dir: Path,
    candidates: list[str],
    max_candidates: int,
    months: list[str],
    fit_months: list[str],
    holdout_month: str,
    top_fracs: list[float],
    match_modes: list[str],
    max_separator_features: int,
    max_separator_features_per_family: int,
    min_separator_auc: float,
    min_separator_abs_gap: float,
    min_direction_consistency: float,
    min_train_rows: int | None,
    min_valid_rows: int | None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_manifest = json.loads((stage_dir / "manifest.json").read_text(encoding="utf-8"))
    candidate_summary = pd.read_csv(stage_dir / "path_aware_label_target_candidate_summary.csv")
    contrast = pd.read_csv(recoverability_dir / "pathgrid_recoverability_feature_contrast.csv")
    selected_candidates = _choose_candidates(candidate_summary, candidates, max_candidates=max_candidates)
    spec_rows = candidate_summary.set_index("candidate").loc[selected_candidates].reset_index()
    specs = [_spec_from_row(row) for _, row in spec_rows.iterrows()]

    reports = stage_manifest.get("reports", {})
    frame, metrics, load_reports = _build_frame(
        labels_path=Path(stage_manifest["labels_path"]),
        feature_dir=Path(stage_manifest.get("feature_dir", DEFAULT_FEATURE_DIR)),
        feature_list_csv=Path(stage_manifest.get("feature_list_csv", DEFAULT_FEATURE_LIST_CSV)),
        max_feature_store_features=stage_manifest.get("max_feature_store_features"),
        include_causal_outcome_priors=bool(reports.get("causal_outcome_priors", {}).get("enabled", False)),
        include_causal_state_path_priors=bool(reports.get("causal_state_path_priors", {}).get("enabled", False)),
        include_event_confirmation_features=bool(reports.get("event_confirmation_features", {}).get("enabled", False)),
        include_adverse_path_composites=bool(reports.get("adverse_path_composites", {}).get("enabled", False)),
        prior_windows_days=[float(v) for v in stage_manifest.get("prior_windows_days", DEFAULT_PRIOR_WINDOWS_DAYS)],
        prior_embargo_hours=float(stage_manifest.get("prior_embargo_hours", 24.0)),
        state_path_prior_features=list(stage_manifest.get("state_path_prior_features", DEFAULT_STATE_PATH_PRIOR_FEATURES)),
        event_feature_store_features=list(stage_manifest.get("event_feature_store_features", DEFAULT_EVENT_FEATURE_STORE_FEATURES)),
    )
    features = _feature_columns(frame)
    month_period = frame["__ts__"].dt.to_period("M").astype(str)
    target_mode = stage_manifest.get("grid", {}).get("target_mode", "hard_binary")
    target_rank_weight = float(stage_manifest.get("grid", {}).get("target_rank_weight", 0.30))
    target_soft_power = float(stage_manifest.get("grid", {}).get("target_soft_power", 1.0))
    min_train = int(min_train_rows if min_train_rows is not None else stage_manifest.get("min_train_rows", 500))
    min_valid = int(min_valid_rows if min_valid_rows is not None else stage_manifest.get("min_valid_rows", 100))

    period_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    separator_feature_rows: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []

    for spec_idx, spec in enumerate(specs, start=1):
        target = _target_for_spec(
            metrics.reset_index(drop=True),
            spec,
            frame["__ts__"].reset_index(drop=True),
            target_mode=target_mode,
            target_rank_weight=target_rank_weight,
            target_soft_power=target_soft_power,
        )
        target.index = frame.index
        for month in months:
            train_mask = month_period.lt(str(month))
            valid_mask = month_period.eq(str(month))
            if int(train_mask.sum()) < min_train or int(valid_mask.sum()) < min_valid:
                continue
            train = frame.loc[train_mask].copy()
            valid_source = frame.loc[valid_mask].copy()
            valid = valid_source.reset_index(drop=True)
            valid_metrics = metrics.loc[valid_mask].copy().reset_index(drop=True)
            valid_target = target.loc[valid_mask].copy().reset_index(drop=True)
            proxy_score, proxy_diag = _proxy_score(
                train=train,
                valid=valid_source,
                features=features,
                target_train=target.loc[train_mask, "target_soft"],
                metrics_train=metrics.loc[train_mask].copy(),
                top_k=int(stage_manifest.get("proxy_top_k", 4)),
                proxy_objective=str(stage_manifest.get("proxy_objective", "economic_ic")),
                min_target_ic=float(stage_manifest.get("proxy_min_target_ic", 0.0)),
                min_utility_ic=float(stage_manifest.get("proxy_min_utility_ic", 0.0)),
                max_bad_mae_ic=float(stage_manifest.get("proxy_max_bad_mae_ic", 0.0)),
                max_wide_ic=float(stage_manifest.get("proxy_max_wide_ic", 0.0)),
                max_timeout_ic=float(stage_manifest.get("proxy_max_timeout_ic", 0.0)),
                utility_weight=float(stage_manifest.get("proxy_utility_weight", 1.0)),
                bad_mae_weight=float(stage_manifest.get("proxy_bad_mae_weight", 1.0)),
                wide_weight=float(stage_manifest.get("proxy_wide_weight", 0.5)),
                timeout_weight=float(stage_manifest.get("proxy_timeout_weight", 0.5)),
            )
            proxy_score = proxy_score.reset_index(drop=True)
            for top_frac in top_fracs:
                selector_scores = {"oracle_target_sort": valid_target["target_soft"].reset_index(drop=True)}
                selector_feature_strings = {"oracle_target_sort": ""}
                selector_scores.update(_selector_scores(proxy_score, pd.Series(np.nan, index=proxy_score.index)))
                selector_feature_strings["economic_ic_proxy_oos"] = ",".join(proxy_diag.get("proxy_features", []))

                for match_mode in match_modes:
                    selected_features = _select_separator_features(
                        contrast,
                        candidate=spec.name,
                        month=str(month),
                        top_frac=float(top_frac),
                        match_mode=str(match_mode),
                        max_features=int(max_separator_features),
                        max_per_family=int(max_separator_features_per_family),
                        min_auc=float(min_separator_auc),
                        min_abs_gap=float(min_separator_abs_gap),
                        min_direction_consistency=float(min_direction_consistency),
                    )
                    if selected_features.empty:
                        continue
                    selected_features = selected_features.copy()
                    selected_features.insert(0, "rank", np.arange(1, len(selected_features) + 1, dtype=np.int64))
                    selected_features.insert(0, "match_mode", str(match_mode))
                    selected_features.insert(0, "top_frac", float(top_frac))
                    selected_features.insert(0, "month", str(month))
                    selected_features.insert(0, "candidate", spec.name)
                    separator_feature_rows.extend(selected_features.to_dict(orient="records"))

                    sep_score = _separator_score(
                        valid=valid,
                        valid_metrics=valid_metrics,
                        selected_features=selected_features,
                        match_mode=str(match_mode),
                    ).reset_index(drop=True)
                    gate_scores = _selector_scores(proxy_score, sep_score)
                    feature_str = ",".join(selected_features["feature"].astype(str).tolist())
                    for name, score in gate_scores.items():
                        if name == "economic_ic_proxy_oos":
                            continue
                        selector_name = f"{name}_{match_mode}"
                        selector_scores[selector_name] = score.reset_index(drop=True)
                        selector_feature_strings[selector_name] = feature_str

                period_rows.extend(
                    _month_period_rows(
                        valid=valid,
                        valid_metrics=valid_metrics,
                        valid_target=valid_target,
                        candidate=spec.name,
                        month=str(month),
                        top_fracs=[float(top_frac)],
                        selector_scores=selector_scores,
                        selector_features=selector_feature_strings,
                    )
                )
                for selector_name, score in selector_scores.items():
                    score = _safe_numeric(score).reset_index(drop=True)
                    selector_features = pd.DataFrame()
                    feature_string = selector_feature_strings.get(selector_name, "")
                    if feature_string:
                        selector_features = pd.DataFrame({"feature": [part for part in feature_string.split(",") if part]})
                    confusion_rows.append(
                        _confusion_row(
                            valid_metrics=valid_metrics,
                            valid_target=valid_target,
                            candidate=spec.name,
                            month=str(month),
                            selector=selector_name,
                            top_frac=float(top_frac),
                            score=score,
                            separator_features=selector_features,
                        )
                    )
                    ledger_rows.extend(
                        _selected_ledger_rows(
                            valid=valid,
                            valid_metrics=valid_metrics,
                            valid_target=valid_target,
                            candidate=spec.name,
                            month=str(month),
                            selector=selector_name,
                            top_frac=float(top_frac),
                            score=score,
                            separator_features=selector_features,
                        )
                    )
        print(json.dumps({"progress": f"{spec_idx}/{len(specs)}", "candidate": spec.name}, sort_keys=True))

    period_frame = pd.DataFrame(period_rows)
    fit_holdout = _candidate_summary(
        _fit_holdout_summary(
            period_frame,
            fit_months=[str(v) for v in fit_months],
            holdout_month=str(holdout_month),
            min_week_rows=10,
            max_timeout_rate=stage_manifest.get("max_timeout_rate"),
        )
    )
    confusion = pd.DataFrame(confusion_rows)
    separator_features = pd.DataFrame(separator_feature_rows)
    ledger = pd.DataFrame(ledger_rows)

    paths = {
        "period_rows": output_dir / "pathgrid_separator_gate_period_rows.csv",
        "fit_holdout": output_dir / "pathgrid_separator_gate_fit_holdout.csv",
        "confusion": output_dir / "pathgrid_separator_gate_confusion.csv",
        "separator_features": output_dir / "pathgrid_separator_gate_features.csv",
        "selected_ledger": output_dir / "pathgrid_separator_gate_selected_ledger.csv",
        "manifest": output_dir / "manifest.json",
    }
    period_frame.to_csv(paths["period_rows"], index=False)
    fit_holdout.to_csv(paths["fit_holdout"], index=False)
    confusion.to_csv(paths["confusion"], index=False)
    separator_features.to_csv(paths["separator_features"], index=False)
    ledger.to_csv(paths["selected_ledger"], index=False)

    manifest = {
        "scope": "pathgrid_separator_gate_ablation",
        "stage_dir": str(stage_dir),
        "recoverability_dir": str(recoverability_dir),
        "output_dir": str(output_dir),
        "labels_path": stage_manifest["labels_path"],
        "rows": int(len(frame)),
        "feature_count": int(len(features)),
        "candidates": [spec.name for spec in specs],
        "candidate_specs": [asdict(spec) for spec in specs],
        "months": [str(v) for v in months],
        "fit_months": [str(v) for v in fit_months],
        "holdout_month": str(holdout_month),
        "top_fracs": [float(v) for v in top_fracs],
        "match_modes": [str(v) for v in match_modes],
        "max_separator_features": int(max_separator_features),
        "max_separator_features_per_family": int(max_separator_features_per_family),
        "min_separator_auc": float(min_separator_auc),
        "min_separator_abs_gap": float(min_separator_abs_gap),
        "min_direction_consistency": float(min_direction_consistency),
        "stage_proxy_objective": stage_manifest.get("proxy_objective"),
        "stage_proxy_top_k": stage_manifest.get("proxy_top_k"),
        "stage_target_mode": target_mode,
        "outputs": {key: str(value) for key, value in paths.items()},
        "load_reports": load_reports,
    }
    markdown = _write_report(
        output_dir=output_dir,
        fit_holdout=fit_holdout,
        period_rows=period_frame,
        confusion=confusion,
        separator_features=separator_features,
        manifest={**manifest, "outputs": {**manifest["outputs"], "markdown": str(output_dir / "pathgrid_separator_gate_ablation.md")}},
    )
    manifest["outputs"]["markdown"] = str(markdown)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-dir", type=Path, default=DEFAULT_STAGE_DIR)
    parser.add_argument("--recoverability-dir", type=Path, default=DEFAULT_RECOVERABILITY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidates", type=lambda value: _parse_csv(value), default="")
    parser.add_argument("--max-candidates", type=int, default=6)
    parser.add_argument("--months", type=lambda value: _parse_csv(value, DEFAULT_MONTHS), default=",".join(DEFAULT_MONTHS))
    parser.add_argument(
        "--fit-months",
        type=lambda value: _parse_csv(value, DEFAULT_FIT_MONTHS),
        default=",".join(DEFAULT_FIT_MONTHS),
    )
    parser.add_argument("--holdout-month", default=DEFAULT_HOLDOUT_MONTH)
    parser.add_argument("--top-fracs", type=_parse_float_csv, default=",".join(str(v) for v in DEFAULT_TOP_FRACS))
    parser.add_argument(
        "--match-modes",
        type=lambda value: _parse_csv(value, DEFAULT_MATCH_MODES),
        default=",".join(DEFAULT_MATCH_MODES),
    )
    parser.add_argument("--max-separator-features", type=int, default=12)
    parser.add_argument("--max-separator-features-per-family", type=int, default=4)
    parser.add_argument("--min-separator-auc", type=float, default=0.70)
    parser.add_argument("--min-separator-abs-gap", type=float, default=0.10)
    parser.add_argument("--min-direction-consistency", type=float, default=0.60)
    parser.add_argument("--min-train-rows", type=int, default=None)
    parser.add_argument("--min-valid-rows", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_report(
        stage_dir=args.stage_dir,
        recoverability_dir=args.recoverability_dir,
        output_dir=args.output_dir,
        candidates=list(args.candidates),
        max_candidates=int(args.max_candidates),
        months=list(args.months),
        fit_months=list(args.fit_months),
        holdout_month=str(args.holdout_month),
        top_fracs=[float(v) for v in args.top_fracs],
        match_modes=list(args.match_modes),
        max_separator_features=int(args.max_separator_features),
        max_separator_features_per_family=int(args.max_separator_features_per_family),
        min_separator_auc=float(args.min_separator_auc),
        min_separator_abs_gap=float(args.min_separator_abs_gap),
        min_direction_consistency=float(args.min_direction_consistency),
        min_train_rows=args.min_train_rows,
        min_valid_rows=args.min_valid_rows,
    )
    print(json.dumps(_json_safe({k: v for k, v in manifest.items() if k != "load_reports"}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
