#!/usr/bin/env python3
"""Causal full-path risk-head ablation for Stage167 selected rows.

This is a diagnostic, not a production training path. It trains a tiny risk
head only on rows already selected by the Stage167 two-head smoke, then chooses
a keep fraction using prior evidence and applies it to the next month.
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
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_first_touch_label_training_smoke import _table  # noqa: E402
from scripts.run_label_quality_proxy_diagnostics import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    DEFAULT_FEATURE_LIST_CSV,
    _json_safe,
    _safe_mean,
    _safe_quantile,
    _spearman,
)
from scripts.run_stage167_selected_fullpath_gate_ablation import (  # noqa: E402
    DEFAULT_LEDGER_CSV,
    _attach_features,
    _load_ledger,
    _metrics,
    _parse_csv,
    _parse_float_csv,
    _safe_numeric,
    _summarize_weekly,
)


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/stage170_stage167_selected_fullpath_risk_head_inner_v1")
DEFAULT_MONTHS = ("2026-04", "2026-05", "2026-06")
DEFAULT_KEEP_FRACS = (0.35, 0.50, 0.65, 0.80, 0.95)
DEFAULT_SEEDS = (42, 7301, 999)


def _parse_int_csv(value: str | list[int] | tuple[int, ...]) -> list[int]:
    if isinstance(value, (list, tuple)):
        return [int(part) for part in value]
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def _full_path_safe_target(
    frame: pd.DataFrame,
    *,
    first_touch_clean_r: float,
    full_path_safe_r: float,
) -> pd.Series:
    return (
        (_safe_numeric(frame["clean_first_touch_exec"]) >= 0.5)
        & (_safe_numeric(frame["first_touch_timeout"]) < 0.5)
        & (_safe_numeric(frame["first_touch_net"]) > 0.0)
        & (_safe_numeric(frame["first_touch_mae_to_sl"]) <= float(first_touch_clean_r))
        & (_safe_numeric(frame["full_path_mae_to_sl"]) <= float(full_path_safe_r))
    ).fillna(False).astype(float)


def _rank_pct(values: Any) -> pd.Series:
    score = _safe_numeric(values)
    if score.notna().sum() == 0:
        return pd.Series(0.5, index=score.index)
    return score.rank(method="average", pct=True).fillna(0.5).clip(0.0, 1.0)


def _top_mask(score: pd.Series, keep_frac: float) -> pd.Series:
    values = _safe_numeric(score).reset_index(drop=True)
    out = pd.Series(False, index=values.index)
    valid = values.notna().to_numpy()
    if not bool(valid.any()):
        return out
    valid_idx = np.flatnonzero(valid)
    k = max(1, int(math.ceil(float(keep_frac) * len(valid_idx))))
    k = min(k, len(valid_idx))
    order = np.argsort(-values.iloc[valid_idx].to_numpy(dtype=np.float64), kind="mergesort")
    out.iloc[valid_idx[order[:k]]] = True
    return out


def _prepare_xy(
    train: pd.DataFrame,
    score_frames: list[pd.DataFrame],
    *,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, list[pd.DataFrame], pd.Series]:
    x_train_raw = train.loc[:, feature_columns].replace([np.inf, -np.inf], np.nan)
    median = x_train_raw.median(numeric_only=True)
    x_train = x_train_raw.fillna(median).fillna(0.0).astype(np.float32, copy=False)
    out_frames: list[pd.DataFrame] = []
    for frame in score_frames:
        x = frame.loc[:, feature_columns].replace([np.inf, -np.inf], np.nan)
        out_frames.append(x.fillna(median).fillna(0.0).astype(np.float32, copy=False))
    return x_train, out_frames, median


def _fit_scores(
    *,
    train: pd.DataFrame,
    score_frames: list[pd.DataFrame],
    feature_columns: list[str],
    seeds: list[int],
    first_touch_clean_r: float,
    full_path_safe_r: float,
) -> list[pd.DataFrame]:
    x_train, x_score_frames, _median = _prepare_xy(train, score_frames, feature_columns=feature_columns)
    y_safe = _full_path_safe_target(
        train,
        first_touch_clean_r=first_touch_clean_r,
        full_path_safe_r=full_path_safe_r,
    )
    y_mae = _safe_numeric(train["full_path_mae_to_sl"]).fillna(10.0).clip(lower=0.0, upper=12.0)

    safe_preds = [np.zeros(len(frame), dtype=np.float64) for frame in score_frames]
    mae_preds = [np.zeros(len(frame), dtype=np.float64) for frame in score_frames]
    class_count = int(y_safe.nunique(dropna=True))

    for seed in seeds:
        if class_count >= 2:
            clf = ExtraTreesClassifier(
                n_estimators=120,
                max_depth=4,
                min_samples_leaf=4,
                max_features="sqrt",
                class_weight="balanced",
                random_state=int(seed),
                n_jobs=1,
            )
            clf.fit(x_train, y_safe.astype(int))
            for idx, x_score in enumerate(x_score_frames):
                proba = clf.predict_proba(x_score)
                classes = list(clf.classes_)
                positive_col = classes.index(1) if 1 in classes else 0
                safe_preds[idx] += proba[:, positive_col]
        else:
            constant = float(y_safe.mean()) if len(y_safe) else 0.0
            for idx in range(len(score_frames)):
                safe_preds[idx] += constant

        reg = ExtraTreesRegressor(
            n_estimators=120,
            max_depth=4,
            min_samples_leaf=4,
            max_features="sqrt",
            random_state=int(seed),
            n_jobs=1,
        )
        reg.fit(x_train, y_mae)
        for idx, x_score in enumerate(x_score_frames):
            mae_preds[idx] += reg.predict(x_score)

    denom = float(max(1, len(seeds)))
    out: list[pd.DataFrame] = []
    for idx, frame in enumerate(score_frames):
        safe_prob = pd.Series(safe_preds[idx] / denom, index=frame.index)
        mae_inverse = pd.Series(-(mae_preds[idx] / denom), index=frame.index)
        blend = (0.70 * _rank_pct(safe_prob)) + (0.30 * _rank_pct(mae_inverse))
        out.append(
            pd.DataFrame(
                {
                    "et_safe_prob": safe_prob.to_numpy(dtype=np.float64, copy=False),
                    "et_mae_inverse": mae_inverse.to_numpy(dtype=np.float64, copy=False),
                    "et_blend_safe_mae": blend.to_numpy(dtype=np.float64, copy=False),
                },
                index=frame.index,
            ),
        )
    return out


def _candidate_row(
    *,
    period: str,
    protocol: str,
    model_name: str,
    keep_frac: float,
    selection_frame: pd.DataFrame,
    holdout_frame: pd.DataFrame,
    selection_score: pd.Series,
    holdout_score: pd.Series,
    selection_baseline: dict[str, Any],
    holdout_baseline: dict[str, Any],
    min_selection_rows: int,
    min_keep_frac: float,
    max_keep_frac: float,
    require_tail_improvement: bool,
    first_touch_clean_r: float,
    full_path_safe_r: float,
    full_path_dirty_r: float,
) -> dict[str, Any]:
    selection_mask = _top_mask(selection_score, keep_frac)
    holdout_mask = _top_mask(holdout_score, keep_frac)
    selected_selection = selection_frame.reset_index(drop=True).loc[selection_mask].copy()
    selected_holdout = holdout_frame.reset_index(drop=True).loc[holdout_mask].copy()
    selection_metrics = _metrics(
        selected_selection,
        first_touch_clean_r=first_touch_clean_r,
        full_path_safe_r=full_path_safe_r,
        full_path_dirty_r=full_path_dirty_r,
    )
    holdout_metrics = _metrics(
        selected_holdout,
        first_touch_clean_r=first_touch_clean_r,
        full_path_safe_r=full_path_safe_r,
        full_path_dirty_r=full_path_dirty_r,
    )
    selection_delta = float(selection_metrics["sum_first_touch_net"]) - float(selection_baseline["sum_first_touch_net"])
    holdout_delta = float(holdout_metrics["sum_first_touch_net"]) - float(holdout_baseline["sum_first_touch_net"])
    tail_ok = True
    if require_tail_improvement:
        tail_ok = (
            float(selection_metrics["p90_full_path_mae_to_sl"]) <= float(selection_baseline["p90_full_path_mae_to_sl"])
            and float(selection_metrics["bad_first_touch_mae_to_sl_rate"])
            <= float(selection_baseline["bad_first_touch_mae_to_sl_rate"]) + 1e-12
            and float(selection_metrics["first_touch_timeout_rate"])
            <= float(selection_baseline["first_touch_timeout_rate"]) + 0.02
        )
    eligible = (
        int(selection_metrics["rows"]) >= int(min_selection_rows)
        and float(keep_frac) >= float(min_keep_frac)
        and float(keep_frac) <= float(max_keep_frac)
        and selection_delta > 0.0
        and bool(tail_ok)
    )
    row: dict[str, Any] = {
        "period": str(period),
        "protocol": str(protocol),
        "model_name": str(model_name),
        "keep_frac": float(keep_frac),
        "selection_delta_sum_first_touch_net": selection_delta,
        "holdout_delta_sum_first_touch_net": holdout_delta,
        "selection_eligible": bool(eligible),
        "selection_score_ic_full_path_safe": _spearman(
            selection_score,
            _full_path_safe_target(
                selection_frame,
                first_touch_clean_r=first_touch_clean_r,
                full_path_safe_r=full_path_safe_r,
            ),
        ),
        "holdout_score_ic_full_path_safe": _spearman(
            holdout_score,
            _full_path_safe_target(
                holdout_frame,
                first_touch_clean_r=first_touch_clean_r,
                full_path_safe_r=full_path_safe_r,
            ),
        ),
    }
    for prefix, metrics in (("selection", selection_metrics), ("holdout", holdout_metrics)):
        for key, value in metrics.items():
            row[f"{prefix}_{key}"] = value
    for prefix, metrics in (("selection_baseline", selection_baseline), ("holdout_baseline", holdout_baseline)):
        for key, value in metrics.items():
            if key in {"rows", "sum_first_touch_net", "bad_full_path_mae_3r_rate", "p90_full_path_mae_to_sl"}:
                row[f"{prefix}_{key}"] = value
    return row


def _baseline_month_rows(
    *,
    period: str,
    holdout: pd.DataFrame,
    first_touch_clean_r: float,
    full_path_safe_r: float,
    full_path_dirty_r: float,
) -> dict[str, Any]:
    metrics = _metrics(
        holdout,
        first_touch_clean_r=first_touch_clean_r,
        full_path_safe_r=full_path_safe_r,
        full_path_dirty_r=full_path_dirty_r,
    )
    return {
        "period": str(period),
        "variant": "stage167_baseline",
        "model_name": "no_gate",
        "keep_frac": 1.0,
        **metrics,
    }


def _risk_head_month_row(
    *,
    period: str,
    selected: pd.Series,
    holdout: pd.DataFrame,
    first_touch_clean_r: float,
    full_path_safe_r: float,
    full_path_dirty_r: float,
) -> dict[str, Any]:
    kept = holdout.reset_index(drop=True).loc[selected.reset_index(drop=True).astype(bool)].copy()
    metrics = _metrics(
        kept,
        first_touch_clean_r=first_touch_clean_r,
        full_path_safe_r=full_path_safe_r,
        full_path_dirty_r=full_path_dirty_r,
    )
    return {
        "period": str(period),
        "variant": "risk_head",
        **metrics,
    }


def _write_markdown(
    *,
    output_dir: Path,
    monthly: pd.DataFrame,
    candidates: pd.DataFrame,
    weekly: pd.DataFrame,
    manifest: dict[str, Any],
) -> Path:
    path = output_dir / "stage170_stage167_selected_fullpath_risk_head.md"
    monthly_cols = [
        "period",
        "variant",
        "model_name",
        "keep_frac",
        "rows",
        "mean_first_touch_net",
        "sum_first_touch_net",
        "bad_full_path_mae_3r_rate",
        "p90_full_path_mae_to_sl",
        "bad_first_touch_mae_to_sl_rate",
        "first_touch_timeout_rate",
        "positive_week_rate",
        "worst_week_first_touch_net",
        "delta_sum_first_touch_net",
    ]
    candidate_cols = [
        "period",
        "protocol",
        "model_name",
        "keep_frac",
        "selected_candidate",
        "selection_eligible",
        "selection_delta_sum_first_touch_net",
        "holdout_delta_sum_first_touch_net",
        "selection_score_ic_full_path_safe",
        "holdout_score_ic_full_path_safe",
        "selection_rows",
        "holdout_rows",
        "selection_bad_full_path_mae_3r_rate",
        "holdout_bad_full_path_mae_3r_rate",
        "selection_p90_full_path_mae_to_sl",
        "holdout_p90_full_path_mae_to_sl",
    ]
    weekly_cols = [
        "period",
        "week",
        "variant",
        "rows",
        "mean_first_touch_net",
        "sum_first_touch_net",
        "bad_full_path_mae_3r_rate",
        "p90_full_path_mae_to_sl",
    ]
    selected = candidates[candidates.get("selected_candidate", False).astype(bool)] if not candidates.empty else candidates
    lines = [
        "# Stage170 Stage167 Selected Full-Path Risk Head",
        "",
        "Scope: causal diagnostic. A tiny ExtraTrees risk head is trained only on prior Stage167 selected rows. Thresholds are keep fractions, not absolute score cutoffs.",
        "",
        f"Protocol: `{manifest['protocol']}`",
        f"Selected ledger: `{manifest['ledger_csv']}`",
        f"Feature dir: `{manifest['feature_dir']}`",
        f"Feature list: `{manifest['feature_list_csv']}`",
        f"Feature count: `{manifest['feature_count']}`",
        f"Keep fractions: `{manifest['keep_fracs']}`",
        f"Require tail improvement on selection frame: `{manifest['require_tail_improvement']}`",
        "",
        "## Monthly Baseline vs Risk Head",
        "",
        _table(monthly, monthly_cols, limit=80),
        "",
        "## Selected Candidate Evidence",
        "",
        _table(selected, candidate_cols, limit=80),
        "",
        "## Weekly Rows",
        "",
        _table(weekly, weekly_cols, limit=120),
        "",
        "## Outputs",
        "",
        f"- Monthly: `{manifest['outputs']['monthly']}`",
        f"- Candidates: `{manifest['outputs']['candidates']}`",
        f"- Weekly: `{manifest['outputs']['weekly']}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_ablation(
    *,
    ledger_csv: Path,
    output_dir: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    max_features: int,
    months: list[str],
    keep_fracs: list[float],
    seeds: list[int],
    protocol: str,
    min_train_rows: int,
    min_selection_rows: int,
    min_keep_frac: float,
    max_keep_frac: float,
    require_tail_improvement: bool,
    first_touch_clean_r: float,
    full_path_safe_r: float,
    full_path_dirty_r: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger = _load_ledger(ledger_csv)
    ledger, feature_manifest, features = _attach_features(
        ledger,
        feature_dir=feature_dir,
        feature_list_csv=feature_list_csv,
        max_gate_features=max_features,
    )
    model_names = ("et_safe_prob", "et_mae_inverse", "et_blend_safe_mae")
    monthly_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    weekly_rows: list[dict[str, Any]] = []

    for period in months:
        prior = ledger[ledger["period"].astype(str) < str(period)].copy()
        holdout = ledger[ledger["period"].astype(str).eq(str(period))].copy()
        if holdout.empty:
            continue
        baseline_row = _baseline_month_rows(
            period=str(period),
            holdout=holdout,
            first_touch_clean_r=first_touch_clean_r,
            full_path_safe_r=full_path_safe_r,
            full_path_dirty_r=full_path_dirty_r,
        )
        baseline_row["delta_sum_first_touch_net"] = 0.0
        monthly_rows.append(baseline_row)
        weekly_rows.extend(_summarize_weekly(holdout, period=str(period), variant="stage167_baseline"))

        prior_months = sorted(prior["period"].astype(str).unique().tolist())
        if protocol == "inner_holdout":
            if len(prior_months) < 2:
                chosen = {
                    "period": str(period),
                    "protocol": str(protocol),
                    "model_name": "pass_through_no_inner_validation",
                    "keep_frac": 1.0,
                    "selected_candidate": True,
                    "selection_eligible": False,
                    "reason": "need_at_least_two_prior_selected_months",
                }
                candidate_rows.append(chosen)
                risk_row = baseline_row.copy()
                risk_row["variant"] = "risk_head"
                risk_row["model_name"] = "pass_through_no_inner_validation"
                monthly_rows.append(risk_row)
                weekly_rows.extend(_summarize_weekly(holdout, period=str(period), variant="risk_head"))
                continue
            selection_month = prior_months[-1]
            train = prior[prior["period"].astype(str) < selection_month].copy()
            selection = prior[prior["period"].astype(str).eq(selection_month)].copy()
        elif protocol == "prior_insample":
            train = prior.copy()
            selection = prior.copy()
        else:
            raise ValueError(f"Unknown protocol: {protocol}")

        if len(train) < int(min_train_rows) or len(selection) < int(min_selection_rows):
            chosen = {
                "period": str(period),
                "protocol": str(protocol),
                "model_name": "pass_through_insufficient_prior_rows",
                "keep_frac": 1.0,
                "selected_candidate": True,
                "selection_eligible": False,
                "train_rows": int(len(train)),
                "selection_rows": int(len(selection)),
                "reason": "insufficient_prior_rows",
            }
            candidate_rows.append(chosen)
            risk_row = baseline_row.copy()
            risk_row["variant"] = "risk_head"
            risk_row["model_name"] = "pass_through_insufficient_prior_rows"
            monthly_rows.append(risk_row)
            weekly_rows.extend(_summarize_weekly(holdout, period=str(period), variant="risk_head"))
            continue

        selection_baseline = _metrics(
            selection,
            first_touch_clean_r=first_touch_clean_r,
            full_path_safe_r=full_path_safe_r,
            full_path_dirty_r=full_path_dirty_r,
        )
        holdout_baseline = _metrics(
            holdout,
            first_touch_clean_r=first_touch_clean_r,
            full_path_safe_r=full_path_safe_r,
            full_path_dirty_r=full_path_dirty_r,
        )
        selection_scores, holdout_scores = _fit_scores(
            train=train,
            score_frames=[selection.reset_index(drop=True), holdout.reset_index(drop=True)],
            feature_columns=features,
            seeds=seeds,
            first_touch_clean_r=first_touch_clean_r,
            full_path_safe_r=full_path_safe_r,
        )

        month_candidates: list[dict[str, Any]] = []
        for model_name in model_names:
            for keep_frac in keep_fracs:
                row = _candidate_row(
                    period=str(period),
                    protocol=protocol,
                    model_name=model_name,
                    keep_frac=float(keep_frac),
                    selection_frame=selection.reset_index(drop=True),
                    holdout_frame=holdout.reset_index(drop=True),
                    selection_score=selection_scores[model_name],
                    holdout_score=holdout_scores[model_name],
                    selection_baseline=selection_baseline,
                    holdout_baseline=holdout_baseline,
                    min_selection_rows=min_selection_rows,
                    min_keep_frac=min_keep_frac,
                    max_keep_frac=max_keep_frac,
                    require_tail_improvement=require_tail_improvement,
                    first_touch_clean_r=first_touch_clean_r,
                    full_path_safe_r=full_path_safe_r,
                    full_path_dirty_r=full_path_dirty_r,
                )
                row["train_rows"] = int(len(train))
                row["selection_rows_available"] = int(len(selection))
                row["holdout_rows_available"] = int(len(holdout))
                month_candidates.append(row)

        candidate_frame = pd.DataFrame(month_candidates)
        eligible = candidate_frame[candidate_frame["selection_eligible"].astype(bool)].copy()
        if eligible.empty:
            selected_candidate = {
                "period": str(period),
                "protocol": str(protocol),
                "model_name": "pass_through_no_positive_selection_delta",
                "keep_frac": 1.0,
                "selected_candidate": True,
                "selection_eligible": False,
                "reason": "no_candidate_improved_selection_frame",
            }
            candidate_rows.extend(month_candidates)
            candidate_rows.append(selected_candidate)
            risk_row = baseline_row.copy()
            risk_row["variant"] = "risk_head"
            risk_row["model_name"] = selected_candidate["model_name"]
            monthly_rows.append(risk_row)
            weekly_rows.extend(_summarize_weekly(holdout, period=str(period), variant="risk_head"))
            continue

        selected = eligible.sort_values(
            [
                "selection_delta_sum_first_touch_net",
                "selection_bad_full_path_mae_3r_rate",
                "selection_p90_full_path_mae_to_sl",
                "keep_frac",
            ],
            ascending=[False, True, True, False],
        ).iloc[0]
        selected_model = str(selected["model_name"])
        selected_keep = float(selected["keep_frac"])

        # Refit on all prior rows after model/keep fraction selection, then apply
        # the selected keep fraction to the holdout month.
        _prior_scores, refit_holdout_scores = _fit_scores(
            train=prior,
            score_frames=[prior.reset_index(drop=True), holdout.reset_index(drop=True)],
            feature_columns=features,
            seeds=seeds,
            first_touch_clean_r=first_touch_clean_r,
            full_path_safe_r=full_path_safe_r,
        )
        holdout_mask = _top_mask(refit_holdout_scores[selected_model], selected_keep)
        risk_row = _risk_head_month_row(
            period=str(period),
            selected=holdout_mask,
            holdout=holdout,
            first_touch_clean_r=first_touch_clean_r,
            full_path_safe_r=full_path_safe_r,
            full_path_dirty_r=full_path_dirty_r,
        )
        risk_row["model_name"] = selected_model
        risk_row["keep_frac"] = selected_keep
        risk_row["delta_sum_first_touch_net"] = float(risk_row["sum_first_touch_net"]) - float(
            baseline_row["sum_first_touch_net"],
        )
        monthly_rows.append(risk_row)
        weekly_rows.extend(
            _summarize_weekly(
                holdout.reset_index(drop=True).loc[holdout_mask].copy(),
                period=str(period),
                variant="risk_head",
            ),
        )
        for row in month_candidates:
            row["selected_candidate"] = (
                row["model_name"] == selected_model and abs(float(row["keep_frac"]) - selected_keep) < 1e-12
            )
            candidate_rows.append(row)

    monthly = pd.DataFrame(monthly_rows)
    candidates = pd.DataFrame(candidate_rows)
    weekly = pd.DataFrame(weekly_rows)
    paths = {
        "monthly": output_dir / "stage170_monthly_baseline_vs_risk_head.csv",
        "candidates": output_dir / "stage170_risk_head_candidates.csv",
        "weekly": output_dir / "stage170_weekly_baseline_vs_risk_head.csv",
        "manifest": output_dir / "manifest.json",
    }
    monthly.to_csv(paths["monthly"], index=False)
    candidates.to_csv(paths["candidates"], index=False)
    weekly.to_csv(paths["weekly"], index=False)
    manifest = {
        "scope": "stage170_stage167_selected_fullpath_risk_head",
        "protocol": str(protocol),
        "ledger_csv": str(ledger_csv),
        "output_dir": str(output_dir),
        "feature_dir": str(feature_dir),
        "feature_list_csv": str(feature_list_csv),
        "feature_store": feature_manifest,
        "feature_count": int(len(features)),
        "features": features,
        "months": list(months),
        "keep_fracs": list(keep_fracs),
        "seeds": list(seeds),
        "model_names": list(model_names),
        "min_train_rows": int(min_train_rows),
        "min_selection_rows": int(min_selection_rows),
        "min_keep_frac": float(min_keep_frac),
        "max_keep_frac": float(max_keep_frac),
        "require_tail_improvement": bool(require_tail_improvement),
        "first_touch_clean_r": float(first_touch_clean_r),
        "full_path_safe_r": float(full_path_safe_r),
        "full_path_dirty_r": float(full_path_dirty_r),
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    markdown = _write_markdown(output_dir=output_dir, monthly=monthly, candidates=candidates, weekly=weekly, manifest=manifest)
    manifest["outputs"]["markdown"] = str(markdown)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-csv", type=Path, default=DEFAULT_LEDGER_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--max-features", type=int, default=80)
    parser.add_argument("--months", default=",".join(DEFAULT_MONTHS))
    parser.add_argument("--keep-fracs", default=",".join(str(v) for v in DEFAULT_KEEP_FRACS))
    parser.add_argument("--seeds", default=",".join(str(v) for v in DEFAULT_SEEDS))
    parser.add_argument("--protocol", choices=("inner_holdout", "prior_insample"), default="inner_holdout")
    parser.add_argument("--min-train-rows", type=int, default=20)
    parser.add_argument("--min-selection-rows", type=int, default=20)
    parser.add_argument("--min-keep-frac", type=float, default=0.35)
    parser.add_argument("--max-keep-frac", type=float, default=0.95)
    parser.add_argument("--require-tail-improvement", action="store_true", default=True)
    parser.add_argument("--no-require-tail-improvement", dest="require_tail_improvement", action="store_false")
    parser.add_argument("--first-touch-clean-r", type=float, default=1.0)
    parser.add_argument("--full-path-safe-r", type=float, default=3.0)
    parser.add_argument("--full-path-dirty-r", type=float, default=3.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_ablation(
        ledger_csv=args.ledger_csv,
        output_dir=args.output_dir,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        max_features=int(args.max_features),
        months=_parse_csv(str(args.months), default=DEFAULT_MONTHS),
        keep_fracs=_parse_float_csv(str(args.keep_fracs)),
        seeds=_parse_int_csv(str(args.seeds)),
        protocol=str(args.protocol),
        min_train_rows=int(args.min_train_rows),
        min_selection_rows=int(args.min_selection_rows),
        min_keep_frac=float(args.min_keep_frac),
        max_keep_frac=float(args.max_keep_frac),
        require_tail_improvement=bool(args.require_tail_improvement),
        first_touch_clean_r=float(args.first_touch_clean_r),
        full_path_safe_r=float(args.full_path_safe_r),
        full_path_dirty_r=float(args.full_path_dirty_r),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
